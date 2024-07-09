from fastapi import APIRouter, HTTPException
from app.models import model, tokenizer, device
from app.schemas.translation import TranslationRequest, TranslationResponse
from app.utils.training import train_model

router = APIRouter()

@router.post("/api/train", response_model=dict)
async def train():
    train_model()
    return {"message": "Training started"}

@router.post("/api/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    input_text = request.text
    inputs = tokenizer.encode("translate Spanish to English: " + input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return TranslationResponse(text=translated_text)
