import os
import requests
import time
import re
import gc
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# --- AYARLAR ---
# Render Environment Variables kısmına HF_TOKEN eklemeyi unutma!
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/akdeniz27/bert-base-turkish-cased-ner"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

class AnalysisRequest(BaseModel):
    text: str

def query_huggingface(text_chunk: str):
    """Deep Learning modelini uzaktan çağırır"""
    payload = {"inputs": text_chunk, "options": {"wait_for_model": True}}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        if response.status_code == 503: # Model uykudaysa
            time.sleep(15)
            return query_huggingface(text_chunk)
        return response.json()
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return []

def clean_name(name: str):
    """BERT tokenlarını temizler (Şey ##da -> Şeyda)"""
    return name.replace(" ", "").replace("##", "").strip()

@app.post("/analyze")
async def analyze_document(data: AnalysisRequest):
    if not data.text:
        return {"error": "Metin bulunamadı"}

    # 1. ADIM: Metni Cümlelere Böl (Context Koruma)
    # Nokta, ünlem veya soru işaretinden sonra bölüyoruz
    sentences = re.split(r'(?<=[.!?])\s+', data.text)
    
    network_map = {}
    entity_frequency = {}

    # 2. ADIM: Cümleleri Gruplayarak İşle (Hugging Face Limiti İçin)
    # Çok fazla API isteği atmamak için cümleleri ~1000 karakterlik paketler yapıyoruz
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 1500:
            current_chunk += " " + sentence
        else:
            # Paketi Analiz Et
            raw_results = query_huggingface(current_chunk)
            
            # Cümle içindeki isimleri ayıkla
            current_entities = []
            if isinstance(raw_results, list):
                for res in raw_results:
                    if res.get('entity_group') == 'PER' or res.get('entity') == 'PER':
                        name = clean_name(res['word'])
                        if len(name) > 1:
                            current_entities.append(name)
                            entity_frequency[name] = entity_frequency.get(name, 0) + 1

            # 3. ADIM: Aynı Paket/Cümle İçindeki Bağları Kur (Weight Toplama)
            unique_entities = list(set(current_entities))
            for i in range(len(unique_entities)):
                for j in range(i + 1, len(unique_entities)):
                    pair = tuple(sorted((unique_entities[i], unique_entities[j])))
                    network_map[pair] = network_map.get(pair, 0) + 1
            
            # Temizlik ve Sıfırlama
            current_chunk = sentence
            gc.collect()

    # 4. ADIM: Sonuçları Tek Bir Dosyada/Obje İçinde Birleştir
    final_network = [
        {"source": p[0], "target": p[1], "weight": w} 
        for p, w in network_map.items()
    ]

    return {
        "status": "success",
        "total_unique_characters": len(entity_frequency),
        "network": final_network,
        "metadata": {
            "model": "BERT-Base-Turkish-NER",
            "engine": "HNA-Inference-Gateway"
        }
    }

@app.get("/")
async def root():
    return {"message": "Hemithea NLP Engine is Running"}
