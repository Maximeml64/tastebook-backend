import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import anthropic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

class ImageItem(BaseModel):
    image_base64: str
    media_type: str = "image/jpeg"

class ScanRequest(BaseModel):
    images: List[ImageItem]

@app.post("/scan-label")
async def scan_label(req: ScanRequest):
    try:
        if not req.images:
            raise HTTPException(status_code=400, detail="No images provided")

        content = []
        for img in req.images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type,
                    "data": img.image_base64,
                },
            })

        content.append({
            "type": "text",
            "text": (
                "Analyse ces etiquettes de vin ou spiritueux (etiquette avant et/ou arriere) "
                "et retourne UNIQUEMENT un objet JSON valide, sans markdown, sans backticks.\n\n"
                "Champs a extraire en combinant les informations des deux etiquettes:\n"
                "- name (string): nom du chateau, domaine ou produit\n"
                "- producer (string): nom du producteur si different du nom\n"
                "- appellation (string): appellation ou region\n"
                "- vintage (number|null): millesime annee 4 chiffres\n"
                "- grape_varieties (array of strings): cepages si mentionnes\n"
                "- type (string): wine ou spirit\n"
                "- color (string): red white rose sparkling si vin null si spiritueux\n"
                "- category (string): categorie si spiritueux null si vin\n\n"
                "Si une information nest pas visible utilise null ou tableau vide.\n"
                "Retourne uniquement le JSON brut rien dautre."
            )
        })

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": content}],
        )

        raw = message.content[0].text.strip()
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()
        data = json.loads(raw)
        return data

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"JSON parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}