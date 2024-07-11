from fastapi import FastAPI
from app.routes import translate

app = FastAPI()

app.include_router(translate.router)
