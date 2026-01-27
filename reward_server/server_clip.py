"""
CLIP Score Server (Port 8162)
Computes cosine similarity between image and text embeddings.
"""
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import uvicorn
from transformers import CLIPProcessor, CLIPModel

# Configuration - set via environment variables
MODEL_PATH = os.environ.get("CLIP_PATH", "")  # Path to clip-vit-large-patch14
PORT = 8162

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
processor = None


class ImageRequest(BaseModel):
    image_base64: str
    prompt: str = ""


def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


@app.on_event("startup")
def load_model():
    global model, processor
    print(f"[CLIP] Loading from {MODEL_PATH}")
    model = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    print("[CLIP] Loaded successfully.")


@app.post("/score")
async def score(req: ImageRequest):
    try:
        image = decode_image(req.image_base64)
        inputs = processor(
            text=[req.prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            clip_score = (text_embeds @ image_embeds.T).item()
            
        return {"score": clip_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
