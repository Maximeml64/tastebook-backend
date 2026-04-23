import os
import base64
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

class ScanRequest(BaseModel):
    image_base64: str
    media_type: str = "image/jpeg"

@app.post("/scan-label")
async def scan_label(req: ScanRequest):
    try:
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": req.media_type,
                                "data": req.image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Analyse cette étiquette de vin ou spiritueux et retourne UNIQUEMENT un objet JSON valide, sans markdown, sans explication.

Champs à extraire :
- name (string) : nom du château, domaine ou produit
- producer (string) : nom du producteur si différent du nom
- appellation (string) : appellation ou région
- vintage (number|null) : millésime (année à 4 chiffres)
- grape_varieties (array of strings) : cépages si mentionnés
- type (string) : "wine" ou "spirit"
- color (string) : "red", "white", "rosé", "sparkling" si c'est un vin, null si spiritueux
- category (string) : catégorie si spiritueux (whisky, rum, cognac, etc.), null si vin

Si une information n'est pas visible sur l'étiquette, utilise null ou tableau vide.
Retourne uniquement le JSON, rien d'autre."""
                        }
                    ],
                }
            ],
        )
        
        raw = message.content[0].text.strip()
        data = json.loads(raw)
        return data
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Impossible de parser la réponse Claude")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}