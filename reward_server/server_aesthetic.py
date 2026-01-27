"""
Aesthetic Score Server (Port 8161)
Uses CLIP features with a linear predictor for aesthetic scoring.
"""
import os
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import uvicorn
from transformers import CLIPProcessor, CLIPModel

# Configuration - set via environment variables
CLIP_PATH = os.environ.get("CLIP_PATH", "")  # Path to clip-vit-large-patch14
AESTHETIC_PATH = os.environ.get("AESTHETIC_PATH", "sa_0_4_vit_l_14_linear.pth")
PORT = 8161

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = None
clip_processor = None
aesthetic_model = None


class AestheticPredictor(nn.Module):
    """Linear predictor on top of CLIP features."""
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 1))

    def forward(self, x):
        return self.layers(x)


class ImageRequest(BaseModel):
    image_base64: str


def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


@app.on_event("startup")
def load_models():
    global clip_model, clip_processor, aesthetic_model
    print(f"[Aesthetic] Loading CLIP from {CLIP_PATH}")
    clip_model = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH, local_files_only=True)
    
    print(f"[Aesthetic] Loading linear weights from {AESTHETIC_PATH}")
    aesthetic_model = AestheticPredictor(768).to(device).eval()
    
    state_dict = torch.load(AESTHETIC_PATH, map_location=device)
    try:
        aesthetic_model.load_state_dict(state_dict)
    except:
        if 'weight' in state_dict and 'bias' in state_dict:
            aesthetic_model.layers[0].weight.data = state_dict['weight']
            aesthetic_model.layers[0].bias.data = state_dict['bias']
    print("[Aesthetic] Loaded successfully.")


@app.post("/score")
async def score(req: ImageRequest):
    try:
        image = decode_image(req.image_base64)
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            aesthetic_score = aesthetic_model(features).item()
            
        return {"score": aesthetic_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
