import stanza
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Türkçe modelini indir ve işlemciyi hazırla
# 'processors="tokenize,ner"' sadece ihtiyacımız olanları yükleyerek RAM kazandırır
stanza.download('tr', processors='tokenize,ner')
nlp = stanza.Pipeline('tr', processors='tokenize,ner', use_gpu=False)

class TextData(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_text(data: TextData):
    if not data.text:
        return []

    # Analizi yap
    doc = nlp(data.text)
    
    # Sadece Kişi (PERSON) etiketlerini ayıkla
    entities = []
    for sent in doc.sentences:
        for ent in sent.ents:
            if ent.type == "PERSON":
                entities.append(ent.text)
    
    # İlişki ağı oluştur
    network_data = []
    for i in range(len(entities) - 1):
        if entities[i] != entities[i+1]:
            network_data.append({
                "source": entities[i],
                "target": entities[i+1],
                "weight": 1
            })
    
    return network_data

@app.get("/")
async def root():
    return {"status": "Stanza NLP Engine is running"}
