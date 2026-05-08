import stanza
from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI()

# RAM tasarrufu için sadece tokenize ve ner modüllerini yüklüyoruz
# use_gpu=False yaparak Render'ın CPU üzerinde stabil çalışmasını sağlıyoruz
try:
    nlp = stanza.Pipeline(
        lang='tr', 
        processors='tokenize,ner', 
        use_gpu=False,
        download_method=None # Model zaten build sırasında indi
    )
except:
    # Eğer build sırasında inmediyse çalışma anında indir (B planı)
    stanza.download('tr', processors='tokenize,ner')
    nlp = stanza.Pipeline(lang='tr', processors='tokenize,ner', use_gpu=False)

class TextData(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_text(data: TextData):
    if not data.text or len(data.text.strip()) == 0:
        return []

    # Metni analiz et
    doc = nlp(data.text)
    
    # Sadece 'PERSON' etiketli varlıkları listele
    entities = []
    for sent in doc.sentences:
        for ent in sent.ents:
            if ent.type == "PERSON":
                entities.append(ent.text)
    
    # Karakterler arasındaki bağı kur
    network_data = []
    for i in range(len(entities) - 1):
        source = entities[i]
        target = entities[i+1]
        
        if source != target:
            network_data.append({
                "source": source,
                "target": target,
                "weight": 1
            })
    
    return network_data

@app.get("/")
async def root():
    return {"status": "Stanza Turkish NER is live!"}
