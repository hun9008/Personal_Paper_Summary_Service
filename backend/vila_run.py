import layoutparser as lp 

from vila.pdftools.pdf_extractor import PDFExtractor
from vila.predictors import HierarchicalPDFPredictor
import matplotlib.pyplot as plt
from collections import defaultdict
import subprocess
import asyncio
import time
from tqdm.asyncio import tqdm_asyncio
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from progress_tracker import progress_status

import re

from elasticsearch import Elasticsearch

es = Elasticsearch("http://elastic.hunian.site:80")

def predict(pdf_path, pdf_extractor, vision_model, layout_model):
    page_tokens, page_images = pdf_extractor.load_tokens_and_image(pdf_path)

    pred_tokens = []
    for page_token, page_image in zip(page_tokens, page_images):
        blocks = vision_model.detect(page_image)
        page_token.annotate(blocks=blocks)
        pdf_data = page_token.to_pagedata().to_dict()
        pred_tokens += layout_model.predict(pdf_data, page_token.page_size)

    return pred_tokens
    

def construct_token_groups(pred_tokens):
    groups, group, group_type, prev_bbox = [], [], None, None
    
    for token in pred_tokens:
        if group_type is None:
            is_continued = True
            
        elif token.type == group_type:
            if group_type == 'section':
                is_continued = abs(prev_bbox[3] - token.coordinates[3]) < 1.
            else:
                is_continued = True

        else:
            is_continued = False

        
        # print(token.text, token.type, is_continued)
        group_type = token.type
        prev_bbox = token.coordinates
        if is_continued:
            group.append(token)
        
        else:
            groups.append(group)
            group = [token]
    
    if group:
        groups.append(group)

    return groups

def join_group_text(group):
    text = ''
    prev_bbox = None
    for token in group:
        if not text:
            text += token.text
    
        else:        
            if abs(prev_bbox[2] - token.coordinates[0]) > 2:
                text += ' ' + token.text
    
            else:
                text += token.text
    
        prev_bbox = token.coordinates
    return text

def construct_section_groups(token_groups):
    section_groups = defaultdict(list)

    section = None
    for group in token_groups:
        group_type = group[0].type
        group_text = join_group_text(group)
        
        if group_type == 'section':
            section = group_text
            section_groups[section]
    
        elif group_type == 'paragraph' and section is not None:
            section_groups[section].append(group_text)

    section_groups = {k: ' '.join(v) for k,v in section_groups.items()}
    return section_groups

async def fetch_ans_llama31(prompt_type: str):
    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            ["ollama", "run", "llama3.1", prompt_type],
            capture_output=True,
            text=True
        )
    )
    
    output = result.stdout.strip()
    return output

async def translate_section(section_groups):
    translated_sections = {}
    for section, text in section_groups.items():
        prompt = f"Translate the following text to Korean (Considering that this is a thesis, important words are used in English.) : \n\n{text}"
        translated_text = await fetch_ans_llama31(prompt)
        translated_sections[section] = translated_text
    return translated_sections

# 함수: LaTeX 수식 앞뒤로 공백 추가
def ensure_closing_font_tag(text: str) -> str:
    # 수식 블록을 감지하고 앞뒤에 공백(빈 줄) 추가
    text = re.sub(r'(?<!\n)\$\$', r'\n$$', text)  # $$ 앞에 빈 줄이 없으면 추가
    text = re.sub(r'\$\$(?!\n)', r'$$\n', text)  # $$ 뒤에 빈 줄이 없으면 추가

    # 문장을 분리하고, <font color="red"> 태그가 닫히지 않은 부분을 감지하여 닫음
    sentences = re.split(r'(?<=\.)\s', text)  
    
    for i, sentence in enumerate(sentences):
        if '<font color="red">' in sentence and '</font>' not in sentence:
            sentences[i] += '</font>'

    # 다시 문장들을 합침
    return ' '.join(sentences)

# 요약 섹션 처리 함수
async def summarize_section(section_groups, update_progress_fn=None, file_name=None):
    summarize_sections = {}
    total_sections = len(section_groups)
    index_name = file_name if file_name else "default_index"
    index_name = index_name.lower().replace(" ", "_").replace(".", "_")
    
    for idx, (section, text) in enumerate(tqdm_asyncio(section_groups.items(), desc="Summarizing sections")):
        prompt = f"""Considering that this is a computer science paper, when using markdown, avoid using # and *, and summarize in paragraphs as much as possible. 
Then, apply LaTeX to the equations and write the most important words or sentences in the summarized part in red.

Restrictions:
1. Sentences starting with <font color='red'> must end with </font>.
2. Before any table (| --- | --- | structure), add an empty line.
3. Output only the summarized results without modifiers.
4. Do not use ----, ====.

Format for each section:

1. **Key Points**: Briefly highlight the most important points of this section.
2. **Summary**: Provide a detailed summary of the content of this section.
3. **Important Equations**: Display any key equations using LaTeX.

$$
\\text{{log loss}} = - \\sum_{{i=1}}^{{n}} y_i \\cdot \\log(p_i) + (1 - y_i) \\cdot \\log(1 - p_i)
$$

4. **Tables**: If there are tables, add a blank line before each table, and format the table in Markdown.

| Example Header 1 | Example Header 2 |
|------------------|------------------|
| Example Data 1   | Example Data 2   |

This is Content : \n\n{text}"""
        
        document = {
            "section": section,  # 섹션 이름
            "text": text         # 섹션 내용
        }

        # ElasticSearch에 데이터 인덱싱 (저장)
        es.index(index=index_name, document=document)

        # translated_text = await fetch_ans_llama31(prompt)

        # # ensure_closing_font_tag 함수로 수식 앞뒤 공백 처리 및 <font> 처리
        # translated_text = ensure_closing_font_tag(translated_text)
        # summarize_sections[section] = translated_text
        summarize_sections[section] = text

        # 진행 상태 업데이트 (50% ~ 90% 구간 차지)
        if update_progress_fn:
            progress = 50 + int((idx / total_sections) * 40)
            update_progress_fn(progress)
    
    return summarize_sections

async def summarize_overall(section_groups, all_file_path):
    
    combined_text = " ".join(section_groups.values())
    print("len : ", len(combined_text))

    save_to_txt(combined_text, section_groups, filename=all_file_path)

    prompt = f"""
    Summarize the entire contents of the paper with a clear and concise structure. Focus on providing an overview that allows readers to understand the main flow and contributions of the paper. Include the following elements:

    1. **Purpose and Research Goals**: What is the main objective of the paper? Summarize the key research questions or problems the paper addresses.
    2. **Background and Motivation**: Provide a brief context of the study. Why is this research important? What background knowledge is necessary to understand the paper's contribution?
    3. **Methodology Overview**: Summarize the methods or approaches used in the paper. Describe any key techniques, models, or algorithms applied to solve the problem.
    4. **Key Contributions and Results**: Highlight the most significant findings or outcomes of the research. Focus on how these results advance the field or provide solutions to the research questions.
    5. **Technical Innovations**: Describe any novel techniques, formulas, or frameworks introduced in the paper. Provide examples of how these innovations are applied.
    6. **Conclusions and Future Work**: Summarize the conclusion of the paper and any suggestions for future research or unresolved questions.

    Provide this summary in a way that gives a complete understanding of the paper's flow and contributions, without diving too deeply into specific numerical details. Make sure the reader can grasp the overall importance and innovation presented in the paper. 

    Here is the text of the paper:\n\n{combined_text}
    """

    overall_summary = await fetch_ans_llama31(prompt)

    # 번역 프롬프트 수정
    prompt = f"""This is a summary of a computer science paper. 
Translate the following summary into Korean while keeping all technical terms, algorithms, and key models in English. 
The explanations can be translated into Korean, but do not use Chinese or Japanese characters.
Please Just Translate Texts, Not Extra Explanation.

Summary to translate:\n\n{overall_summary}"""
    overall_summary = await fetch_ans_llama31(prompt)
        
    return overall_summary

# 마크다운 파일로 저장
def save_to_md(overall_summary, section_summaries, filename="summary.md"):
    md_content = f"# Document Summary\n\n**Overall Summary:**\n\n{overall_summary}\n\n"

    if section_summaries != "":
        for section, summary in section_summaries.items():
            md_content += f"## {section}\n\n{summary}\n\n"

    # 마크다운 파일로 저장
    with open(filename, "w", encoding="utf-8") as f:
        f.write(md_content)

def save_to_txt(overall_text, section_texts, filename="all_raw_text.txt"):
    txt_content = f"# Document Summary\n\n**Overall Summary:**\n\n{overall_text}\n\n"
    for section, text in section_texts.items():
        txt_content += f"## {section}\n\n{text}\n\n"

    # 텍스트 파일로 저장
    with open(filename, "w", encoding="utf-8") as f:
        f.write(txt_content)