import os
import requests
import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Token'ı Render Environment'tan güvenli bir şekilde çekiyoruz
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/akdeniz27/bert-base-turkish-cased-ner"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class AnalysisRequest(BaseModel):
    text: str

def query_huggingface(payload):
    # Eğer token tanımlanmamışsa hata verelim
    if not HF_TOKEN:
        return {"error": "HF_TOKEN bulunamadı! Lütfen Render Environment ayarlarına ekleyin."}
        
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 503:
        time.sleep(15)
        return query_huggingface(payload)
    
    return response.json()

@app.post("/analyze")
async def process_nlp(data: AnalysisRequest):
    # ... (Geri kalan karakter ağı oluşturma logic'i aynı kalıyor)
    raw_results = query_huggingface({"inputs": data.text, "options": {"wait_for_model": True}})
    
    # ... Karakter ağı işleme kodların buraya gelecek ...
    return {"status": "Success", "data": raw_results}
