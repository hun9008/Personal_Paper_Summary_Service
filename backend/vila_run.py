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

pdf_extractor = PDFExtractor("pdfplumber")
page_tokens, page_images = pdf_extractor.load_tokens_and_image("./Neural Collaborative Filtering.pdf")

vision_model = lp.EfficientDetLayoutModel("lp://PubLayNet")  
pdf_predictor = HierarchicalPDFPredictor.from_pretrained("allenai/hvila-block-layoutlm-finetuned-docbank")

pred_tokens = predict("./Neural Collaborative Filtering.pdf", pdf_extractor, vision_model, pdf_predictor)
token_groups = construct_token_groups(pred_tokens)
section_groups = construct_section_groups(token_groups)

sections = list(section_groups.keys())
# print(sections)

# print(section_groups[sections[0]])

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
async def summarize_section(section_groups, update_progress_fn=None):
    summarize_sections = {}
    total_sections = len(section_groups)
    
    for idx, (section, text) in enumerate(tqdm_asyncio(section_groups.items(), desc="Summarizing sections")):
        prompt = f"""Considering that this is a computer science paper, when using markdown, avoid using # and *, and summarize in paragraphs as much as possible. Then, apply LaTeX to the equations and write the most important words or sentences in the summarized part in red.

Restrictions:
1. Sentences starting with <font color='red'> must end with </font>.
2. Output only the summarized results without modifiers.
3. Do not use ----, ====.
This is Content : \n\n{text}"""
        
        translated_text = await fetch_ans_llama31(prompt)

        # ensure_closing_font_tag 함수로 수식 앞뒤 공백 처리 및 <font> 처리
        translated_text = ensure_closing_font_tag(translated_text)
        summarize_sections[section] = translated_text

        # 진행 상태 업데이트 (50% ~ 90% 구간 차지)
        if update_progress_fn:
            progress = 50 + int((idx / total_sections) * 40)
            update_progress_fn(progress)
    
    return summarize_sections


async def summarize_overall(section_groups):
    
    combined_text = " ".join(section_groups.values())
    print("len : ", len(combined_text))

    prompt = f"Summarize the entire contents of the paper Just print the contents. :\n\n{combined_text}"
    overall_summary = await fetch_ans_llama31(prompt)

    prompt = f"This is a summary of a computer science paper. Please translate it into Korean. The terms used in the paper should be kept in English, and only the explanations should be translated into Korean. Chinese characters and Japanese should not be used. Please print only the content. : \n\n{overall_summary}"
    overall_summary = await fetch_ans_llama31(prompt)
        
    return overall_summary

# 마크다운 파일로 저장
def save_to_md(overall_summary, section_summaries, filename="summary.md"):
    md_content = f"# Document Summary\n\n**Overall Summary:**\n\n{overall_summary}\n\n"
    for section, summary in section_summaries.items():
        md_content += f"## {section}\n\n{summary}\n\n"

    # 마크다운 파일로 저장
    with open(filename, "w", encoding="utf-8") as f:
        f.write(md_content)

async def main():
    # 예시로 섹션 그룹 가져오기
    start_time = time.time()

    pred_tokens = predict("./Neural Collaborative Filtering.pdf", pdf_extractor, vision_model, pdf_predictor)
    token_groups = construct_token_groups(pred_tokens)
    section_groups = construct_section_groups(token_groups)

    # 섹션 요약
    # section_summaries = await translate_section(section_groups)
    section_summaries = await summarize_section(section_groups)

    # 전체 섹션 요약으로 최종 요약 생성
    overall_summary = await summarize_overall(section_groups)

    # 마크다운 파일로 저장
    save_to_md(overall_summary, section_summaries)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

# 프로그램 실행
# asyncio.run(main())