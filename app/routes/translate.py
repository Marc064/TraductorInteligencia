from fastapi import APIRouter
from pydantic import BaseModel
from app.models import model, tokenizer, device
from app.utils.training import train_model

router = APIRouter()

class TranslationRequest(BaseModel):
    text: str

@router.post("/api/train", response_model=dict)
async def train():
    train_model()
    return {"message": "Training susses"}

@router.post("/translate/")
async def translate_text(request: TranslationRequest):
    input_text = "translate Spanish to English: " + request.text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translated_text": translated_text}
