from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import os
import shutil
import asyncio
from vila_run import predict, construct_token_groups, construct_section_groups, summarize_section, summarize_overall, save_to_md, save_to_txt, fetch_ans_llama31
from vila.pdftools.pdf_extractor import PDFExtractor
from vila.predictors import HierarchicalPDFPredictor
import layoutparser as lp
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError
import boto3
import time
from typing import AsyncGenerator
from progress_tracker import progress_status, update_progress

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# AWS S3 설
AWS_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# 파일을 저장하는 함수
async def save_file(file: UploadFile, upload_dir: str):
    os.makedirs(upload_dir, exist_ok=True)  # 폴더가 없으면 생성
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path

def read_summary_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        global progress_status
        progress_status["progress"] = 0  # 진행률 초기화

        # PDF 파일 이름을 기반으로 폴더 경로 생성
        file_base_name = os.path.splitext(file.filename)[0]  # 확장자 제외 파일 이름
        folder_path = os.path.join("./uploads", file_base_name)  # uploads/파일이름 폴더
        summary_filepath = os.path.join(folder_path, "summary.md")  # 요약 결과 파일 경로
        all_text_filepath = os.path.join(folder_path, "all_text.txt")  # 전체 텍스트 파일 경로  

        # 만약 summary.md 파일이 이미 존재하면, 바로 반환
        if os.path.exists(summary_filepath):
            summary_content = read_summary_file(summary_filepath)
            progress_status["progress"] = 100  # 이미 완료된 파일
            return JSONResponse(content={"summary_content": summary_content})

        # 그렇지 않다면, 폴더를 만들고 PDF 파일 저장
        file_path = await save_file(file, folder_path)
        progress_status["progress"] = 10  # 파일 저장 완료, 진행률 10%

        # vila_run.py에 정의된 함수를 사용하여 PDF 처리
        pdf_extractor = PDFExtractor("pdfplumber")
        vision_model = lp.EfficientDetLayoutModel("lp://PubLayNet")
        pdf_predictor = HierarchicalPDFPredictor.from_pretrained("allenai/hvila-block-layoutlm-finetuned-docbank")

        # Predict 함수 실행
        pred_tokens = predict(file_path, pdf_extractor, vision_model, pdf_predictor)
        progress_status["progress"] = 30  # 예측 완료, 진행률 30%

        token_groups = construct_token_groups(pred_tokens)
        section_groups = construct_section_groups(token_groups)
        progress_status["progress"] = 50  # 섹션 그룹화 완료, 진행률 50%

        # 섹션 요약
        section_summaries = await summarize_section(section_groups, update_progress, file_base_name)  # 진행 상태 콜백 함수 전달
        progress_status["progress"] = 90  # 섹션 요약 완료, 진행률 90%

        # 전체 요약 생성
        overall_summary = await summarize_overall(section_groups, all_text_filepath)
        progress_status["progress"] = 95  # 전체 요약 완료, 진행률 95%

        # 마크다운 파일로 저장
        save_to_md(overall_summary, "", filename=summary_filepath)

        # summary.md 파일의 모든 내용을 읽어서 반환
        summary_content = read_summary_file(summary_filepath)
        progress_status["progress"] = 100  # 완료, 진행률 100%

        # 결과 반환 (summary.md의 내용을 JSON으로 반환)
        return JSONResponse(content={"summary_content": summary_content})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
def upload_file_to_s3(file_path, s3_file_name):
    try:
        # S3에 파일 업로드
        s3_client.upload_file(file_path, AWS_BUCKET_NAME, s3_file_name)

        # S3에서 접근 가능한 파일 URL 생성
        file_url = f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{s3_file_name}"
        return file_url

    except FileNotFoundError:
        print("The file was not found")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None

@app.get("/get-summary")
async def get_summary(file_name: str = Query(...)):
    try:
        # S3나 로컬에서 PDF와 요약 파일을 가져오는 로직이 포함되어 있다고 가정
        uploads_folder = f"./uploads/{file_name}"  # 업로드 경로
        pdf_filename = f"{file_name}.pdf"
        summary_filename = "summary.md"

        pdf_filepath = os.path.join(uploads_folder, pdf_filename)
        summary_filepath = os.path.join(uploads_folder, summary_filename)

        if not os.path.exists(summary_filepath):
            return JSONResponse(content={"error": "Summary not found"}, status_code=404)

        # 요약 파일 읽기
        with open(summary_filepath, "r", encoding="utf-8") as f:
            summary_content = f.read()

        # S3에서 가져온 PDF URL을 반환하는 대신 로컬 파일을 웹에서 바로 열 수 있게 헤더를 추가
        return JSONResponse(content={"pdf_url": f"http://localhost:8000/get-pdf?file_name={file_name}", "summary_content": summary_content})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get-pdf")
async def get_pdf(file_name: str = Query(...)):
    uploads_folder = f"./uploads/{file_name}"
    pdf_filename = f"{file_name}.pdf"
    pdf_filepath = os.path.join(uploads_folder, pdf_filename)

    if os.path.exists(pdf_filepath):
        # 브라우저에서 바로 PDF를 열 수 있도록 inline 설정
        return FileResponse(pdf_filepath, media_type="application/pdf", headers={"Content-Disposition": "inline"})
    else:
        return JSONResponse(content={"error": "PDF not found"}, status_code=404)
    
@app.get("/get-prev-summary")
async def get_prev_summary():
    try:
        uploads_folder = "./uploads"
        if not os.path.exists(uploads_folder):
            return JSONResponse(content={"error": "No uploads found"}, status_code=404)

        # 요약이 완료된 논문 폴더들 목록
        completed_summaries = []
        
        # uploads 폴더 안의 각 폴더를 확인
        for folder_name in os.listdir(uploads_folder):
            folder_path = os.path.join(uploads_folder, folder_name)
            summary_filepath = os.path.join(folder_path, "summary.md")
            
            # summary.md 파일이 있는 폴더만 리스트에 추가
            if os.path.exists(summary_filepath):
                completed_summaries.append(folder_name)

        # 요약이 완료된 폴더가 없으면 빈 배열 반환
        if not completed_summaries:
            return JSONResponse(content={"message": "No completed summaries found"}, status_code=200)

        return JSONResponse(content={"completed_summaries": completed_summaries}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

async def generate_progress() -> AsyncGenerator[str, None]:
    last_progress = progress_status["progress"]
    last_update_time = time.time()  # 마지막 업데이트 시간을 기록
    
    while progress_status["progress"] < 100:
        current_progress = progress_status["progress"]
        current_time = time.time()

        # 진행 상태가 1초 이상 동안 변화가 없으면 0.001% 증가
        if current_progress == last_progress and current_time - last_update_time > 1:
            progress_status["progress"] += 0.001
            current_progress = progress_status["progress"]

        # 진행 상황 전송
        yield f"data: {current_progress:.3f}\n\n"  # 소수점 3자리까지 전송
        
        # 진행 상태 및 시간 업데이트
        last_progress = current_progress
        last_update_time = current_time
        
        await asyncio.sleep(1)  # 1초마다 진행 상황 전송

    # 마지막으로 100% 전송
    yield f"data: 100\n\n"

@app.get("/progress")
async def progress():
    return StreamingResponse(generate_progress(), media_type="text/event-stream")

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()  # 요청 본문에서 JSON 데이터를 추출
        prompt = data.get("prompt")  # 'prompt' 키로부터 값을 가져옴

        if not prompt:
            return JSONResponse(content={"error": "No prompt provided"}, status_code=400)

        response = await fetch_ans_llama31(prompt)
        return JSONResponse(content={"response": response})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)