from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 이름 (Hugging Face 모델 허브에서 확인)
model_name = "meta-llama/Llama-3.2-1B"

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # GPU 자동 할당
    torch_dtype="auto"  # 데이터 타입 자동 결정
)
