"""
ImageReward Score Server (Port 8165)
Image quality reward scoring based on BLIP architecture.
"""
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import uvicorn
import glob
import ImageReward as RM

# Configuration - set via environment variables
MODEL_PATH = os.environ.get("IMAGEREWARD_PATH", "")  # Path to ImageReward model
PORT = 8165

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


class ImageRequest(BaseModel):
    image_base64: str
    prompt: str


def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


@app.on_event("startup")
def load_model():
    global model
    print(f"[ImageReward] Loading from {MODEL_PATH}")
    
    try:
        from ImageReward.ImageReward import ImageReward
        
        # Find .pt file
        state_dict_path = os.path.join(MODEL_PATH, "ImageReward.pt")
        if not os.path.exists(state_dict_path):
            pts = glob.glob(os.path.join(MODEL_PATH, "*.pt"))
            if pts:
                state_dict_path = pts[0]
            else:
                raise FileNotFoundError("No .pt file found")
        
        # Find config
        med_config_path = os.path.join(MODEL_PATH, "med_config.json")
        if not os.path.exists(med_config_path):
            import ImageReward
            pkg_path = os.path.dirname(ImageReward.__file__)
            med_config_path = os.path.join(pkg_path, "models", "BLIP", "configs", "med_config.json")
        
        model = ImageReward(device=device, med_config=med_config_path)
        model.load_state_dict(torch.load(state_dict_path, map_location=device), strict=False)
        model.eval()
        print("[ImageReward] Loaded successfully.")
    except Exception as e:
        print(f"[ImageReward] Failed to load: {e}")
        raise e


@app.post("/score")
async def score(req: ImageRequest):
    try:
        image = decode_image(req.image_base64)
        
        with torch.no_grad():
            reward = model.score(req.prompt, image)
            
        return {"score": reward}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
