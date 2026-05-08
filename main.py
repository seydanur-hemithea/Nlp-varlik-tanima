from fastapi import FastAPI
from transformers import pipeline
import pandas as pd

app = FastAPI()

# Türkçe için eğitilmiş XLM-RoBERTa tabanlı Varlık Tanıma (NER) modeli
# Bu model SpaCy'den çok daha modern ve güçlüdür.
ner_pipeline = pipeline("ner", model="akdeniz27/bert-base-turkish-cased-ner", aggregation_strategy="simple")

@app.post("/analyze")
async def analyze_text(data: dict):
    text = data.get("text", "")
    if not text:
        return []

    # Analiz
    results = ner_pipeline(text)
    
    # Sadece PER (Kişi) etiketlerini alıyoruz
    entities = [res['word'] for res in results if res['entity_group'] == 'PER']
    
    # Karakter ağını oluştur
    network_data = []
    for i in range(len(entities) - 1):
        network_data.append({
            "source": entities[i],
            "target": entities[i+1],
            "weight": 1
        })
    
    return network_data
