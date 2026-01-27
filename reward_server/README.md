# Reward Server

Reward model servers for evaluating inpainting quality.

## Overview

FastAPI-based servers providing various image quality metrics via HTTP API.

## Files

| File | Port | Description |
|------|------|-------------|
| `server_aesthetic.py` | 8161 | Aesthetic score using CLIP + linear predictor |
| `server_clip.py` | 8162 | CLIP text-image similarity score |
| `server_hpsv2.py` | 8163 | Human Preference Score v2 |
| `server_pickscore.py` | 8166 | PickScore preference scoring |
| `bound.py` | - | Boundary smoothness calculator (local, not a server) |

## Environment Variables

Set model paths before starting servers:

```bash
export CLIP_PATH="/path/to/clip-vit-large-patch14"
export HPSV2_PATH="/path/to/HPSv2"
export PICKSCORE_PATH="/path/to/PickScore_v1"
```

## Usage

### Start Servers

```bash
# Aesthetic server
python server_aesthetic.py

# CLIP server
python server_clip.py

# HPSv2 server
python server_hpsv2.py

# PickScore server
python server_pickscore.py
```

### API Endpoints

All servers expose a `/score` POST endpoint accepting JSON:

```json
{
    "image_base64": "<base64 encoded image>",
    "prompt": "text description (optional for aesthetic)"
}
```

Response:
```json
{
    "score": 0.85
}
```

### Example Client

```python
import base64
import requests

with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8162/score",
    json={"image_base64": img_b64, "prompt": "a beautiful sunset"}
)
print(response.json()["score"])
```

## Port Mapping

| Service | Port |
|---------|------|
| Aesthetic | 8161 |
| CLIP | 8162 |
| HPSv2 | 8163 |
| PickScore | 8166 |
