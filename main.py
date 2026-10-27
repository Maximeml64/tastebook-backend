import os
import json
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import anthropic
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

# Garde-fous de coût (surchargeables par env var Railway).
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(8 * 1024 * 1024)))
MAX_IMAGE_BASE64_CHARS = int(os.environ.get("MAX_IMAGE_BASE64_CHARS", str(5 * 1024 * 1024)))
MAX_IMAGES_PER_REQUEST = int(os.environ.get("MAX_IMAGES_PER_REQUEST", "4"))
IP_HOURLY_LIMIT = int(os.environ.get("IP_HOURLY_LIMIT", "120"))
GLOBAL_DAILY_CAP = int(os.environ.get("GLOBAL_DAILY_CAP", "1500"))

IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
DOCUMENT_MEDIA_TYPES = {"application/pdf"}


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not x_api_key or x_api_key != BACKEND_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"detail": f"Request body too large (max {MAX_BODY_BYTES} bytes)"})
    return await call_next(request)


class ImageItem(BaseModel):
    image_base64: str
    media_type: str = "image/jpeg"


class ScanRequest(BaseModel):
    images: List[ImageItem]


def validate_images(images: List[ImageItem], allowed_media_types: set):
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")
    if len(images) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Too many images (max {MAX_IMAGES_PER_REQUEST})")
    for img in images:
        if img.media_type not in allowed_media_types:
            raise HTTPException(status_code=400, detail=f"Unsupported media_type {img.media_type}")
        if not img.image_base64 or len(img.image_base64) > MAX_IMAGE_BASE64_CHARS:
            raise HTTPException(status_code=413, detail=f"Image too large (max {MAX_IMAGE_BASE64_CHARS} base64 chars)")


# In-memory rate limiters (keyed by endpoint then by device / ip / global)
_calls: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
_calls_lock = threading.Lock()


def _consume(bucket_name: str, key: str, max_calls: int, window_seconds: int, status: int, message: str):
    now = datetime.now(timezone.utc).timestamp()
    with _calls_lock:
        bucket = _calls[bucket_name]
        calls = [t for t in bucket[key] if now - t < window_seconds]
        if len(calls) >= max_calls:
            retry_after_seconds = int(window_seconds - (now - min(calls)))
            bucket[key] = calls
            raise HTTPException(status_code=status, detail=message, headers={"Retry-After": str(retry_after_seconds)})
        calls.append(now)
        bucket[key] = calls


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_rate_limits(endpoint: str, request: Request, device_id: str, max_per_day: int):
    # 1. Fusible global : protège la facture Anthropic même si les identifiants
    #    d'appareil sont falsifiés.
    _consume("global", "all", GLOBAL_DAILY_CAP, 86400, 503, "Service temporarily unavailable (daily capacity reached)")
    # 2. Par IP (horaire) : l'identifiant d'appareil est choisi par le client, l'IP non.
    _consume("ip", client_ip(request), IP_HOURLY_LIMIT, 3600, 429, f"Rate limit exceeded: max {IP_HOURLY_LIMIT} requests per hour.")
    # 3. Par appareil (quotidien) : limite d'usage normale, visible par l'app.
    _consume(endpoint, device_id, max_per_day, 86400, 429, f"Rate limit exceeded: max {max_per_day} scans per 24h.")


def first_text(message) -> str:
    for block in message.content:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            return block.text.strip()
    raise HTTPException(status_code=502, detail="Empty model response")


def parse_json_text(raw: str):
    lines = [l for l in raw.split("\n") if not l.startswith("```")]
    return json.loads("\n".join(lines).strip())


@app.post("/scan-label", dependencies=[Depends(verify_api_key)])
async def scan_label(req: ScanRequest, request: Request, x_device_id: Optional[str] = Header(default=None)):
    if not x_device_id:
        raise HTTPException(status_code=400, detail="Missing X-Device-Id header")

    validate_images(req.images, IMAGE_MEDIA_TYPES)
    check_rate_limits("scan-label", request, x_device_id, max_per_day=50)

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
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": content}],
        )

        return parse_json_text(first_text(message))

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"JSON parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan-invoice", dependencies=[Depends(verify_api_key)])
async def scan_invoice(req: ScanRequest, request: Request, x_device_id: Optional[str] = Header(default=None)):
    if not x_device_id:
        raise HTTPException(status_code=400, detail="Missing X-Device-Id header")

    validate_images(req.images, IMAGE_MEDIA_TYPES | DOCUMENT_MEDIA_TYPES)
    check_rate_limits("scan-invoice", request, x_device_id, max_per_day=20)

    try:
        content = []
        for img in req.images:
            if img.media_type == "application/pdf":
                # Bloc "document" requis par l'API Claude pour un PDF (Haiku 4.5
                # accepte le PDF base64 sans beta header, jusqu'a 100 pages).
                content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": img.image_base64,
                    },
                })
            else:
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

        return parse_json_text(first_text(message))

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"JSON parse error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GenerateRecipesRequest(BaseModel):
    # Liste d'ingredients que l'utilisateur a chez lui (ex: ["poulet", "tomate", "riz"])
    ingredients: List[str]
    # Nombre de portions souhaite (par defaut 4). Borne par le serveur.
    servings: Optional[int] = 4
    # Nombre de recettes a generer (par defaut 3, max 5).
    count: Optional[int] = 3


@app.post("/generate-recipes", dependencies=[Depends(verify_api_key)])
async def generate_recipes(req: GenerateRecipesRequest, request: Request, x_device_id: Optional[str] = Header(default=None)):
    """
    Genere des recettes a partir d'une liste d'ingredients (feature 'qu'est-ce que j'ai dans mon frigo').
    Retourne du JSON brut pret a etre injecte dans le store front.
    """
    if not x_device_id:
        raise HTTPException(status_code=400, detail="Missing X-Device-Id header")

    # Validation defensive de l'input pour eviter les abus / prompts trop longs
    if not req.ingredients:
        raise HTTPException(status_code=400, detail="No ingredients provided")
    if len(req.ingredients) > 30:
        raise HTTPException(status_code=400, detail="Too many ingredients (max 30)")
    cleaned = [str(i).strip()[:60] for i in req.ingredients if str(i).strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="Empty ingredients list after sanitisation")

    servings = max(1, min(req.servings or 4, 12))
    count = max(1, min(req.count or 3, 5))

    check_rate_limits("generate-recipes", request, x_device_id, max_per_day=5)

    ingredients_str = ", ".join(cleaned)

    prompt_text = (
        f"L'utilisateur a ces ingredients chez lui : {ingredients_str}.\n\n"
        f"Propose {count} recettes francaises faisables en utilisant un MAXIMUM de ces ingredients. "
        f"Chaque recette est pour {servings} personnes.\n\n"
        "Reponds UNIQUEMENT avec un objet JSON valide, sans markdown, sans backticks, sans texte avant ou apres.\n\n"
        "Schema strict attendu :\n"
        "{\n"
        "  \"recipes\": [\n"
        "    {\n"
        "      \"title\": string (nom court de la recette en francais),\n"
        "      \"category\": string (UNE parmi 'entree', 'plat', 'dessert', 'autre'),\n"
        "      \"difficulty\": string (UNE parmi 'easy', 'medium', 'hard'),\n"
        "      \"tags\": array of strings (max 4 tags pertinents en francais minuscule, ex 'vege', 'rapide', 'sans gluten'),\n"
        "      \"prepMinutes\": number (temps preparation en minutes),\n"
        "      \"cookMinutes\": number (temps cuisson en minutes),\n"
        "      \"baseServings\": number (= " + str(servings) + "),\n"
        "      \"ingredients\": [\n"
        "        { \"name\": string (nom ingredient), \"quantity\": number|null, \"unit\": string (ex 'g', 'ml', 'cuillere a soupe', 'piece', '' si pas pertinent) }\n"
        "      ],\n"
        "      \"steps\": array of strings (etapes claires et concises, 3 a 8 etapes),\n"
        "      \"matched\": array of strings (ingredients du frigo utilises dans cette recette),\n"
        "      \"missing\": array of strings (ingredients de la recette qui ne sont PAS dans le frigo, donc a acheter)\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Regles strictes :\n"
        "- Privilegie les recettes qui utilisent le PLUS d'ingredients du frigo et le MOINS d'achats.\n"
        "- Pas d'ingredient invente : si une recette necessite quelque chose de farfelu, choisis-en une autre.\n"
        "- Quantites realistes pour " + str(servings) + " personnes.\n"
        "- Etapes claires et actionnables, en francais, sans numerotation (la numerotation est ajoutee par l'app).\n"
        "- Les tags sont des mots-cles courts en minuscule (sans diacritiques de preference).\n"
        "- Si un champ quantity n'a pas de sens (ex 'sel'), mets null.\n"
        "- Retourne uniquement le JSON brut, rien d'autre."
    )

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=3072,
            messages=[{"role": "user", "content": prompt_text}],
        )

        data = parse_json_text(first_text(message))

        # Validation legere de la structure pour eviter d'envoyer des donnees pourries au front
        if not isinstance(data, dict) or not isinstance(data.get("recipes"), list):
            raise HTTPException(status_code=422, detail="Malformed LLM response (no recipes array)")

        return data

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"JSON parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
