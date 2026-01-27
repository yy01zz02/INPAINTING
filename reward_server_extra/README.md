# Reward Server Extra

Additional reward model servers (separate environment from main reward_server/).

## Overview

Contains servers that require different Python environments or dependencies from the main reward_server directory.

## Directories

### hpsv3/
Human Preference Score v3 server (Port 8164)

### IR/
ImageReward server (Port 8165)

## Environment Variables

```bash
export HPSV3_PATH="/path/to/HPSv3.safetensors"
export IMAGEREWARD_PATH="/path/to/ImageReward"
```

## Usage

```bash
# HPSv3 server
cd hpsv3
python server_hpsv3.py

# ImageReward server
cd IR
python server_imagereward.py
```

## Port Mapping

| Service | Port |
|---------|------|
| HPSv3 | 8164 |
| ImageReward | 8165 |
