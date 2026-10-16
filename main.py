from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import chromadb
import json
from pathlib import Path
from uuid import uuid4

app = FastAPI(title="AnimMate Studio")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

OLLAMA_URL = "http://localhost:11434/api/generate"

class PromptRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
def read_index():
    with (BASE_DIR / "templates" / "index.html").open("r", encoding="utf-8") as f:
        return f.read()

@app.post("/generate-animation")
def generate_animation(req: PromptRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must not be empty")

    chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
    collection = chroma_client.get_or_create_collection(name="animation_memory")

    system_prompt = (
        "You are an animation director. Create a lively, coherent story from the user's idea in exactly 12 scenes. "
        "Use move_right, bounce, and scale in that repeating order, starting with move_right in scene 1. "
        "Dialogue must be one short spoken sentence with no quotation marks, scene labels, stage directions, or JSON fragments. "
        "Return only the requested JSON."
    )

    scene_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["scene_number", "background", "character", "action", "dialogue"],
        "properties": {
            "scene_number": {"type": "integer", "enum": list(range(1, 13))},
            "background": {"type": "string"},
            "character": {"type": "string"},
            "action": {"type": "string", "enum": ["move_right", "bounce", "scale"]},
            "dialogue": {
                "type": "string",
                "maxLength": 160,
                "description": "One short spoken sentence only.",
            },
        },
    }
    output_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "scenes"],
        "properties": {
            "title": {"type": "string"},
            "scenes": {
                "type": "array",
                "minItems": 12,
                "maxItems": 12,
                "items": scene_schema,
            },
        },
    }

    payload = {
        "model": "gemma3:1b",
        "system": system_prompt,
        "prompt": f"Animation idea: {prompt}",
        "stream": False,
        "format": output_schema,
        "options": {
            "temperature": 0.2,
            "repeat_penalty": 1.15,
            "num_predict": 1536,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except requests.Timeout as exc:
        raise HTTPException(status_code=504, detail="Animation generation timed out") from exc
    except requests.ConnectionError as exc:
        raise HTTPException(status_code=503, detail="Could not connect to Ollama") from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Ollama request failed") from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {response.status_code}",
        ) from exc

    try:
        res_data = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Ollama returned invalid JSON") from exc

    raw_text = res_data.get("response") if isinstance(res_data, dict) else None
    if not isinstance(raw_text, str):
        raise HTTPException(status_code=502, detail="Ollama response is missing generated text")

    try:
        script_data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Ollama generated invalid script JSON") from exc

    scenes = script_data.get("scenes") if isinstance(script_data, dict) else None
    valid_scenes = (
        isinstance(scenes, list)
        and len(scenes) == 12
        and all(
            isinstance(scene, dict)
            and isinstance(scene.get("background"), str)
            and isinstance(scene.get("character"), str)
            and isinstance(scene.get("action"), str)
            and scene.get("action") in {"move_right", "bounce", "scale"}
            and isinstance(scene.get("dialogue"), str)
            for scene in scenes
        )
    )
    if not isinstance(script_data, dict) or not isinstance(script_data.get("title"), str) or not valid_scenes:
        raise HTTPException(status_code=502, detail="Ollama generated a script with an invalid structure")

    for scene_number, scene in enumerate(scenes, start=1):
        scene["scene_number"] = scene_number

    collection.add(
        documents=[prompt],
        metadatas=[{"title": script_data["title"]}],
        ids=[str(uuid4())],
    )

    return script_data