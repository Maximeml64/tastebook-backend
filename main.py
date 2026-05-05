import os
import json
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import anthropic
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from datetime import datetime, timezone
from collections import defaultdict
import threading

app = FastAPI()

# CORS strict : pas d'appels depuis un navigateur. L'app mobile RN
# n'envoie pas de header Origin, donc CORS ne la concerne pas.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_methods=["POST"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Auth par API key custom dans le header X-API-Key.
# Configure BACKEND_API_KEY dans les env vars Railway.
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY")
if not BACKEND_API_KEY:
    raise RuntimeError("BACKEND_API_KEY env var is required")


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not x_api_key or x_api_key != BACKEND_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class ImageItem(BaseModel):
    image_base64: str
    media_type: str = "image/jpeg"


class ScanRequest(BaseModel):
    images: List[ImageItem]


# In-memory rate limiter for /scan-invoice (per device, not per IP)
_device_calls: dict[str, list[float]] = defaultdict(list)
_device_calls_lock = threading.Lock()


def check_invoice_rate_limit(device_id: str, max_per_day: int = 20):
    now = datetime.now(timezone.utc).timestamp()
    window = 86400  # 24h
    with _device_calls_lock:
        calls = _device_calls[device_id]
        calls = [t for t in calls if now - t < window]
        if len(calls) >= max_per_day:
            oldest = min(calls)
            retry_after_seconds = int(window - (now - oldest))
            _device_calls[device_id] = calls
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: max {max_per_day} scans per 24h. Retry after {retry_after_seconds}s.",
                headers={"Retry-After": str(retry_after_seconds)}
            )
        calls.append(now)
        _device_calls[device_id] = calls


@app.post("/scan-label", dependencies=[Depends(verify_api_key)])
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


@app.post("/scan-invoice", dependencies=[Depends(verify_api_key)])
async def scan_invoice(req: ScanRequest, x_device_id: Optional[str] = Header(default=None)):
    if not x_device_id:
        raise HTTPException(status_code=400, detail="Missing X-Device-Id header")

    check_invoice_rate_limit(x_device_id, max_per_day=20)

    if not req.images:
        raise HTTPException(status_code=400, detail="No images provided")

    try:
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
                "Analyse cette facture francaise d'equipement domestique ou de service "
                "et retourne UNIQUEMENT un objet JSON valide, sans markdown, sans backticks.\n\n"
                "Schema attendu :\n"
                "{\n"
                "  \"vendor_name\": string|null,\n"
                "  \"vendor_address\": string|null,\n"
                "  \"vendor_siret\": string|null,\n"
                "  \"invoice_number\": string|null,\n"
                "  \"purchase_date\": string format YYYY-MM-DD ou null,\n"
                "  \"item_name\": string|null (nom court de l'equipement principal, ex 'Pompe a chaleur Daikin Altherma 3'),\n"
                "  \"brand\": string|null,\n"
                "  \"model\": string|null,\n"
                "  \"serial_number\": string|null,\n"
                "  \"category_suggestion\": string|null (UNE valeur parmi: realestate, car, moto, bike, scooter, boiler, ac, heatpump, waterheater, energy, pool, appliance, garden, multimedia, security, pet, other),\n"
                "  \"total_ttc\": number|null (en euros, format decimal),\n"
                "  \"total_ht\": number|null,\n"
                "  \"vat_amount\": number|null,\n"
                "  \"warranty_years\": number|null (extrait du texte si mentionne),\n"
                "  \"payment_method\": string|null,\n"
                "  \"notes\": string|null (informations supplementaires utiles, ex 'installation incluse', 'maintenance annuelle')\n"
                "}\n\n"
                "Regles strictes :\n"
                "- Utilise null pour toute info manquante, jamais une chaine vide ni 'N/A'.\n"
                "- Les prix au format decimal (1234.56) sans symbole, sans separateur de milliers.\n"
                "- La date au format ISO YYYY-MM-DD.\n"
                "- Pour category_suggestion, choisis la valeur la plus pertinente parmi la liste exacte fournie. Exemples : pompe a chaleur -> heatpump, chaudiere -> boiler, climatisation -> ac, lave-linge -> appliance, panneaux solaires -> energy.\n"
                "- Si plusieurs items sur la facture, concentre-toi sur l'item principal (le plus cher ou le plus volumineux).\n"
                "- Retourne uniquement le JSON brut, rien d'autre."
            )
        })

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
