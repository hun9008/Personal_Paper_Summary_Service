from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline
import torch

# Hugging Face에서 모델 로드
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 모델과 토크나이저 로드
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id)

# GPU 사용 가능 시 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 예시 입력
prompt = "The LLaMA 3.1 model is designed for"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 모델 추론
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)

# 결과 디코딩
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)