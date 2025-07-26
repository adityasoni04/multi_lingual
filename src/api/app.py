# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
from ..core.processor import Processor

class Task(str, Enum):
    translate = "Translate"
    transliterate = "Transliterate"

class ProcessRequest(BaseModel):
    text: str
    task: Task
    source_lang: str = Field(..., example="Hindi")
    target_lang: str = Field(..., example="English")

class ProcessResponse(BaseModel):
    result: str

app = FastAPI(title="Massively Multilingual Agent API")

MODEL_PATH = "./models/google-flan-t5-xl-multilingual"
processor = Processor(model_path=MODEL_PATH)

@app.post("/process", response_model=ProcessResponse)
def process_text(request: ProcessRequest):
    instruction = f"{request.task.value} {request.source_lang} to {request.target_lang}: {request.text}"
    print(f"Received instruction: {instruction}")
    try:
        result_text = processor.process(instruction)
        return {"result": result_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Massively Multilingual Agent API!"}