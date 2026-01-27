"""
HPSv3 Score Server (Port 8164)
Human Preference Score v3 based on MizzenAI model.
"""
import os
import sys

# Add HPSv3 repo path to allow direct import if cloned locally
current_dir = os.path.dirname(os.path.abspath(__file__))
hpsv3_repo_path = os.path.join(current_dir, "HPSv3")
if os.path.exists(hpsv3_repo_path) and hpsv3_repo_path not in sys.path:
    sys.path.append(hpsv3_repo_path)

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import uvicorn
import tempfile
from hpsv3 import HPSv3RewardInferencer

# Configuration - set via environment variables
MODEL_PATH = os.environ.get("HPSV3_PATH", "")  # Path to HPSv3.safetensors
PORT = 8164

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
inferencer = None


class ImageRequest(BaseModel):
    image_base64: str
    prompt: str


def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


@app.on_event("startup")
def load_model():
    global inferencer
    print(f"[HPSv3] Loading from {MODEL_PATH}")
    
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        inferencer = HPSv3RewardInferencer(checkpoint_path=MODEL_PATH, device=device)
    else:
        inferencer = HPSv3RewardInferencer(device=device)
    print("[HPSv3] Loaded successfully.")


@app.post("/score")
async def score(req: ImageRequest):
    try:
        image = decode_image(req.image_base64)
        
        # HPSv3 requires file path, use temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name
        
        try:
            result = inferencer.infer(tmp_path, req.prompt)
            hps_score = result['human_preference_score']
        finally:
            os.unlink(tmp_path)
            
        return {"score": hps_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)

