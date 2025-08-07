import os
import sys
import tempfile
import uuid
import glob
import io
import base64
import asyncio
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.append("../stable_diffusion")

sys.path.append("..")
from edit_cli import load_model_from_config, CFGDenoiser
from omegaconf import OmegaConf
import k_diffusion as K
import einops
from einops import rearrange
from torch import autocast

# Global model cache - each model has its own loaded checkpoint
model_cache = {}  # Structure: {model_name: {model, model_wrap, model_wrap_cfg, null_token, checkpoint_path}}
config = None
current_model = None  # Track the current model being used

def print_flush(*args, **kwargs):
    """Print with immediate flush to ensure output appears immediately"""
    print(*args, **kwargs, flush=True)

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    # Remove data URL prefix if present
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 95) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


# Available models configuration
AVAILABLE_MODELS = {
    "fashionpix2pix-ft": {
        "name": "FashionPix2Pix FT",
        "description": "Fashion-trained InstructPix2Pix model with timestamp",
        "checkpoint_dir": "../logs/fashion-training/train_fashion_fashion-8gpu-training-20250727_000556/checkpoints/trainstep_checkpoints/",
        "default_checkpoint": "epoch=000066-step=000042000.ckpt"
    },
    "fashionpix2pix-full": {
        "name": "FashionPix2Pix Full", 
        "description": "Fashion-trained InstructPix2Pix model full training",
        "checkpoint_dir": "../logs/fashion-training/train_fashion_fashion-8gpu-training/checkpoints/trainstep_checkpoints/",
        "default_checkpoint": None  # Will use latest available
    },
    "instructpix2pix": {
        "name": "InstructPix2Pix Original",
        "description": "Original pre-trained InstructPix2Pix model",
        "checkpoint_dir": "../checkpoints/",
        "default_checkpoint": "instruct-pix2pix-00-22000.ckpt"
    }
}

# Configuration
DEFAULT_CONFIG = "../configs/generate.yaml"
DEFAULT_MODEL = "fashionpix2pix-ft"
DEFAULT_RESOLUTION = 512
DEFAULT_STEPS = 50
DEFAULT_CFG_TEXT = 7.5
DEFAULT_CFG_IMAGE = 1.5

# Request/Response models
class EditRequest(BaseModel):
    image: str  # base64 encoded image
    instruction: str
    model: Optional[str] = None
    checkpoint: Optional[str] = None
    resolution: int = DEFAULT_RESOLUTION
    steps: int = DEFAULT_STEPS
    cfg_text: float = DEFAULT_CFG_TEXT
    cfg_image: float = DEFAULT_CFG_IMAGE
    seed: Optional[int] = None

class EditResponse(BaseModel):
    success: bool
    image: Optional[str] = None  # base64 encoded image
    seed: Optional[int] = None
    instruction: Optional[str] = None
    model: Optional[str] = None
    checkpoint: Optional[str] = None
    error: Optional[str] = None

def load_model_checkpoint(model_name: str, checkpoint_path: str):
    """Load a specific checkpoint for a specific model into cache"""
    global model_cache, config
    
    # Clear existing model from cache if it exists
    if model_name in model_cache:
        old_cache = model_cache[model_name]
        del old_cache['model'], old_cache['model_wrap'], old_cache['model_wrap_cfg']
        torch.cuda.empty_cache()
    
    # Load config if not loaded
    if config is None:
        config = OmegaConf.load(DEFAULT_CONFIG)
    
    # Load the model
    model = load_model_from_config(config, checkpoint_path)
    model.eval().cuda()
    
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    
    # Cache the loaded model
    model_cache[model_name] = {
        'model': model,
        'model_wrap': model_wrap,
        'model_wrap_cfg': model_wrap_cfg,
        'null_token': null_token,
        'checkpoint_path': checkpoint_path
    }
    
    # Removed duplicate success message - already logged in load_all_default_models

def get_latest_checkpoint_for_model(model_name: str) -> str:
    """Get the latest/default checkpoint path for a model"""
    model_info = get_model_info(model_name)
    
    if model_info["default_checkpoint"]:
        checkpoint_path = os.path.join(model_info["checkpoint_dir"], model_info["default_checkpoint"])
        if os.path.exists(checkpoint_path):
            return checkpoint_path
    
    # Use latest checkpoint
    checkpoints = get_available_checkpoints_for_model(model_name)
    if checkpoints:
        return checkpoints[-1]["path"]
    
    raise ValueError(f"No checkpoints found for model: {model_name}")

def load_all_default_models():
    """Load default/latest checkpoint for all available models"""
    print_flush(f"📦 Loading {len(AVAILABLE_MODELS)} models: {', '.join(AVAILABLE_MODELS.keys())}")
    
    total_models = len(AVAILABLE_MODELS)
    loaded_count = 0
    
    for i, model_name in enumerate(AVAILABLE_MODELS.keys(), 1):
        print_flush("=" * 50)
        print_flush(f"⏳ [{i}/{total_models}] Loading {model_name}...")
        try:
            checkpoint_path = get_latest_checkpoint_for_model(model_name)
            load_model_checkpoint(model_name, checkpoint_path)
            print_flush(f"✅ [{i}/{total_models}] {model_name}: {os.path.basename(checkpoint_path)} loaded")
            loaded_count += 1
        except Exception as e:
            print_flush(f"❌ [{i}/{total_models}] {model_name}: Failed to load - {e}")
            import traceback
            traceback.print_exc()
    
    print_flush("=" * 50)
    if loaded_count == total_models:
        print_flush(f"🎉 All {total_models} models loaded successfully!")
    else:
        print_flush(f"⚠️  Loaded {loaded_count}/{total_models} models into cache")
    print_flush("=" * 50)

def ensure_model_loaded(model_name: str = None, checkpoint_path: str = None):
    """Ensure specified model with checkpoint is loaded in cache"""
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    # Check if model is in cache
    if model_name not in model_cache:
        # Model not loaded, load default checkpoint
        if checkpoint_path is None:
            checkpoint_path = get_latest_checkpoint_for_model(model_name)
        print_flush(f"📥 Loading {model_name} with checkpoint: {os.path.basename(checkpoint_path)}")
        load_model_checkpoint(model_name, checkpoint_path)
        return
    
    # Model is in cache, check if checkpoint needs to be swapped
    if checkpoint_path is not None:
        current_checkpoint = model_cache[model_name]['checkpoint_path']
        if current_checkpoint != checkpoint_path:
            print_flush(f"🔄 Switching checkpoint for {model_name}: {os.path.basename(current_checkpoint)} → {os.path.basename(checkpoint_path)}")
            load_model_checkpoint(model_name, checkpoint_path)
            print_flush(f"✅ Checkpoint switch completed for {model_name}")
        else:
            print_flush(f"✓ Using cached {model_name} with checkpoint: {os.path.basename(current_checkpoint)}")

def get_model_from_cache(model_name: str):
    """Get model components from cache"""
    if model_name not in model_cache:
        raise ValueError(f"Model {model_name} not loaded in cache")
    
    cache = model_cache[model_name]
    return cache['model'], cache['model_wrap_cfg'], cache['null_token']

def get_model_info(model_name: str) -> Dict:
    """Get model configuration by name"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(AVAILABLE_MODELS.keys())}")
    return AVAILABLE_MODELS[model_name]

def get_checkpoint_path(checkpoint_name: str, model_name: str = None) -> str:
    """Convert checkpoint name to full path for specified model"""
    if checkpoint_name.startswith("/") or checkpoint_name.startswith("../"):
        return checkpoint_name  # Already full path
    
    # Use current model or default
    if model_name is None:
        model_name = current_model or DEFAULT_MODEL
    
    model_info = get_model_info(model_name)
    checkpoint_dir = model_info["checkpoint_dir"]
    
    # If it's just a filename, construct full path
    if checkpoint_name.endswith(".ckpt"):
        return os.path.join(checkpoint_dir, checkpoint_name)
    
    return checkpoint_name

def get_available_checkpoints_for_model(model_name: str) -> List[Dict]:
    """Get all available checkpoints for a specific model"""
    model_info = get_model_info(model_name)
    checkpoint_dir = model_info["checkpoint_dir"]
    
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoint_pattern = os.path.join(checkpoint_dir, "*.ckpt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    checkpoints = []
    for ckpt_path in sorted(checkpoint_files):
        filename = os.path.basename(ckpt_path)
        
        # Parse filename to extract epoch and step (if available)
        epoch, step = None, None
        if filename.startswith("epoch=") and "-step=" in filename:
            parts = filename.replace(".ckpt", "").split("-step=")
            epoch_part = parts[0].replace("epoch=", "")
            step_part = parts[1]
            
            # Handle versions like "000001000-v1" by taking only the number part
            if "-" in step_part:
                step_part = step_part.split("-")[0]
            
            try:
                epoch, step = int(epoch_part), int(step_part)
            except ValueError:
                # Skip files with unparseable epoch/step format
                epoch, step = None, None
        
        # Get file stats
        stat = os.stat(ckpt_path)
        file_size = stat.st_size
        modified_time = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        checkpoints.append({
            "filename": filename,
            "path": ckpt_path,
            "epoch": epoch,
            "step": step,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "modified_time": modified_time
        })
    
    # Sort by step number if available, otherwise by filename
    checkpoints.sort(key=lambda x: x["step"] if x["step"] is not None else x["filename"])
    return checkpoints

def process_image(
    input_image: Image.Image,
    edit_instruction: str,
    model_name: str = DEFAULT_MODEL,
    resolution: int = DEFAULT_RESOLUTION,
    steps: int = DEFAULT_STEPS,
    cfg_text: float = DEFAULT_CFG_TEXT,
    cfg_image: float = DEFAULT_CFG_IMAGE,
    seed: Optional[int] = None
) -> Image.Image:
    """Process image with edit instruction using specified model"""
    
    # Get model from cache
    model, model_wrap_cfg, null_token = get_model_from_cache(model_name)
    
    # Resize image
    width, height = input_image.size
    factor = resolution / max(width, height)
    factor = np.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
    
    if seed is None:
        seed = torch.randint(0, 100000, (1,)).item()
    
    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        # Prepare conditioning
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([edit_instruction])]
        input_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_tensor = rearrange(input_tensor, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_tensor).mode()]
        
        # Prepare unconditioning
        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]
        
        # Generate
        model_wrap = model_cache[model_name]['model_wrap']
        sigmas = model_wrap.get_sigmas(steps)
        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": cfg_text,
            "image_cfg_scale": cfg_image,
        }
        
        torch.manual_seed(seed)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        
        # Decode
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    
    return edited_image, seed

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    print_flush("🚀 recreategoods API Server starting...")
    print_flush("📦 Loading all model checkpoints...")
    try:
        load_all_default_models()
        print_flush("✅ Server startup complete - all models loaded!")
    except Exception as e:
        print_flush(f"⚠️  Warning: Error during model loading: {e}")
        print_flush("Models will be loaded on first request instead.")
    
    yield  # Server is running
    
    # Shutdown (optional cleanup)
    print_flush("🛑 Server shutting down...")

app = FastAPI(
    title="recreategoods API",
    description="API for fashion image editing using InstructPix2Pix",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "recreategoods API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "models_loaded": len(model_cache),
        "loaded_models": {
            model_name: {
                "checkpoint": os.path.basename(cache['checkpoint_path']),
                "full_path": cache['checkpoint_path']
            }
            for model_name, cache in model_cache.items()
        },
        "ready_for_inference": len(model_cache) > 0,
        "available_models": list(AVAILABLE_MODELS.keys())
    }

@app.post("/edit", response_model=EditResponse)
async def edit_image(request: EditRequest):
    """Edit an image based on text instruction using JSON request/response with base64 images"""
    
    try:
        # Determine model to use
        target_model = request.model or DEFAULT_MODEL
        
        # Ensure model is loaded
        if target_model not in AVAILABLE_MODELS:
            return EditResponse(
                success=False,
                error=f"Unknown model: {target_model}. Available: {list(AVAILABLE_MODELS.keys())}"
            )
        
        checkpoint_path = None
        if request.checkpoint:
            checkpoint_path = get_checkpoint_path(request.checkpoint, target_model)
            if not os.path.exists(checkpoint_path):
                return EditResponse(
                    success=False,
                    error=f"Checkpoint not found: {request.checkpoint}"
                )
        
        ensure_model_loaded(target_model, checkpoint_path)
        
        # Log checkpoint usage for this edit request
        current_checkpoint = model_cache[target_model]['checkpoint_path']
        print_flush(f"🎨 Processing edit request - Model: {target_model}, Checkpoint: {os.path.basename(current_checkpoint)}, Instruction: '{request.instruction}'")
        
        # Track processing time
        import time
        start_time = time.time()
        
        # Convert base64 to image
        input_image = base64_to_image(request.image)
        
        # Process the image
        edited_image, used_seed = process_image(
            input_image=input_image,
            edit_instruction=request.instruction,
            model_name=target_model,
            resolution=request.resolution,
            steps=request.steps,
            cfg_text=request.cfg_text,
            cfg_image=request.cfg_image,
            seed=request.seed
        )
        
        # Convert result to base64
        result_base64 = image_to_base64(edited_image)
        
        # Log successful completion with timing
        processing_time = time.time() - start_time
        print_flush(f"✅ Edit completed - Model: {target_model}, Checkpoint: {os.path.basename(current_checkpoint)}, Time: {processing_time:.2f}s, Seed: {used_seed}")
        
        return EditResponse(
            success=True,
            image=result_base64,
            seed=used_seed,
            instruction=request.instruction,
            model=target_model,
            checkpoint=os.path.basename(model_cache[target_model]['checkpoint_path']) if target_model in model_cache else "default"
        )
        
    except Exception as e:
        # Log error with context
        error_msg = f"Processing error: {str(e)}"
        if 'target_model' in locals() and 'current_checkpoint' in locals():
            print_flush(f"❌ Edit failed - Model: {target_model}, Checkpoint: {os.path.basename(current_checkpoint)}, Error: {error_msg}")
        else:
            print_flush(f"❌ Edit failed - Error: {error_msg}")
        
        return EditResponse(
            success=False,
            error=error_msg
        )

@app.get("/models")
async def get_available_models():
    """Get list of all available models"""
    return {
        "models": {
            model_id: {
                "id": model_id,
                "name": info["name"],
                "description": info["description"],
                "checkpoint_dir": info["checkpoint_dir"],
                "default_checkpoint": info["default_checkpoint"],
                "is_current": model_id == (current_model or DEFAULT_MODEL)
            }
            for model_id, info in AVAILABLE_MODELS.items()
        },
        "current_model": current_model or DEFAULT_MODEL,
        "default_model": DEFAULT_MODEL
    }

@app.get("/checkpoints")
async def get_available_checkpoints(
    model: Optional[str] = None
):
    """Get list of all available checkpoints for specified model"""
    
    if model is None:
        model = current_model or DEFAULT_MODEL
    
    if model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    try:
        checkpoints = get_available_checkpoints_for_model(model)
        model_info = get_model_info(model)
        
        # Mark current checkpoint
        current_checkpoint_path = model_cache.get(model, {}).get('checkpoint_path')
        for checkpoint in checkpoints:
            checkpoint["is_current"] = (checkpoint["path"] == current_checkpoint_path)
        
        return {
            "model": model,
            "model_info": model_info,
            "total_checkpoints": len(checkpoints),
            "current_checkpoint": current_checkpoint_path,
            "checkpoints": checkpoints
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading checkpoints: {str(e)}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind to")
    parser.add_argument("--workers", default=1, type=int, help="Number of worker processes")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False
    )