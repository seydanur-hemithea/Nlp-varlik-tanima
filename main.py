from fastapi import FastAPI
import spacy
import pandas as pd

app = FastAPI()

# Türkçe Large modelini yüklüyoruz (CPU üzerinde en iyi performans için)
nlp = spacy.load("tr_core_news_lg")

@app.post("/analyze")
async def analyze_text(data: dict):
    text = data.get("text", "")
    if not text:
        return []

    doc = nlp(text)
    
    # Sadece Kişi (PERSON) etiketli varlıkları çıkarıyoruz
    entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    # Karakterler arası bağ kurma (Co-occurrence)
    # Basit mantık: Aynı paragraf/blok içinde ardışık geçen isimleri bağla
    network_data = []
    for i in range(len(entities) - 1):
        network_data.append({
            "source": entities[i],
            "target": entities[i+1],
            "weight": 1
        })
    
    return network_data
