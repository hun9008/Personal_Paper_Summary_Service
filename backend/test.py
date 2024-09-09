import subprocess
import asyncio
import pdfplumber
import time

# PDF에서 텍스트를 추출하는 함수
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_with_abstract_handling(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        abstract_found = False  # Abstract 찾기 전까지 좌우 구분 없이 처리
        
        for page in pdf.pages:
            if abstract_found:
                # 좌우 구분해서 처리하는 로직
                width = page.width
                height = page.height

                # 페이지의 왼쪽 절반 (좌측 텍스트)
                left_bbox = (0, 0, width / 2, height)
                left_page_text = page.within_bbox(left_bbox).extract_text()

                # 페이지의 오른쪽 절반 (우측 텍스트)
                right_bbox = (width / 2, 0, width, height)
                right_page_text = page.within_bbox(right_bbox).extract_text()

                # 좌우 텍스트를 합쳐서 하나의 페이지 텍스트로 결합
                page_text = ""
                if left_page_text:
                    page_text += left_page_text + " "
                if right_page_text:
                    page_text += right_page_text

                # 각 페이지의 텍스트를 추가
                full_text += page_text + "\n"

            else:
                # Abstract 이전까지는 좌우 구분 없이 전체 텍스트를 추출
                page_text = page.extract_text()
                if "Abstract" in page_text or "ABSTRACT" in page_text:
                    abstract_found = True  # Abstract를 찾으면 이후부터는 좌우 구분 처리
                    continue
                full_text += page_text + "\n"

    return full_text

# Llama 3.1에 질의하는 함수
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

# PDF 전체를 요약하는 함수
async def summarize_all_pdf(pdf_path):
    # PDF에서 좌우 텍스트를 페이지별로 추출
    pdf_text = extract_text_with_abstract_handling(pdf_path)

    # 추출된 텍스트를 .txt로 저장
    with open("pdf_text.txt", "w", encoding="utf-8") as f:
        f.write(pdf_text)
    
    # 요약할 프롬프트 생성
    prompt = f"Summarize the following document (Please summarize in Korean):\n\n{pdf_text}"
    summary = await fetch_ans_llama31(prompt)
    return summary


# 메인 함수 (실행 시간 측정 포함)
async def main(pdf_path):
    start_time = time.time()  # 시작 시간 기록
    print("Processing started...")

    summary = await summarize_all_pdf(pdf_path)

    # 요약 결과를 summary.md 파일에 저장
    with open("summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Summary of {pdf_path}\n\n")
        f.write(summary)

    end_time = time.time()  # 끝나는 시간 기록
    elapsed_time = end_time - start_time  # 소요 시간 계산

    print(f"\nSummary of the PDF has been saved to summary.md.")
    print(f"Processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    pdf_path = "Neural Collaborative Filtering.pdf"
    asyncio.run(main(pdf_path))