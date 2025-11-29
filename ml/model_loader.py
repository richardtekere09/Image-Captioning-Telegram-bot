"""
Model Loader Module - BLIP Model with Singleton Pattern
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Tuple, Optional
import logging
from config import settings

logger = logging.getLogger(__name__)

_model_instance: Optional[Tuple[BlipForConditionalGeneration, BlipProcessor]] = None


class ModelLoadError(Exception):
    """Raised when model cannot be loaded."""

    pass


def get_model() -> Tuple[BlipForConditionalGeneration, BlipProcessor]:
    """
    Get BLIP model and processor (Singleton pattern).

    Returns model instantly after first load.

    Returns:
        Tuple of (model, processor)

    Raises:
        ModelLoadError: If model cannot be loaded
    """
    global _model_instance

    if _model_instance is not None:
        logger.debug("Returning cached model instance")
        return _model_instance

    logger.info("Loading model for the first time...")
    _model_instance = load_blip_model()
    logger.info("Model loaded and cached successfully")

    return _model_instance


def load_blip_model() -> Tuple[BlipForConditionalGeneration, BlipProcessor]:
    """
    Download and load BLIP model from Hugging Face.

    Returns:
        Tuple of (model, processor)

    Raises:
        ModelLoadError: If loading fails
    """
    try:
        device = _get_device()
        logger.info(f"Using device: {device}")

        # Load processor
        logger.info(f"Loading processor for {settings.MODEL_NAME}...")
        processor = BlipProcessor.from_pretrained(
            settings.MODEL_NAME, cache_dir=settings.CACHE_DIR
        )
        logger.info("✅ Processor loaded")

        # Load model
        logger.info(f"Loading model {settings.MODEL_NAME}...")
        logger.info("(First time: downloading ~990 MB, please wait...)")

        model = BlipForConditionalGeneration.from_pretrained(
            settings.MODEL_NAME,
            cache_dir=settings.CACHE_DIR,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        logger.info("✅ Model loaded")

        # Move to device
        logger.info(f"Moving model to {device}...")
        model = model.to(device)
        model.eval()
        logger.info("✅ Model ready")

        # Log GPU memory if applicable
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"📊 GPU Memory: {allocated:.2f} GB")

        return model, processor

    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(error_msg)
        raise ModelLoadError(error_msg) from e


def _get_device() -> str:
    """
    Determine device to use (GPU or CPU).

    Returns:
        Device string: "cuda" or "cpu"

    Raises:
        ModelLoadError: If CUDA requested but not available
    """
    config_device = settings.DEVICE.lower()

    if config_device == "cuda":
        if not torch.cuda.is_available():
            raise ModelLoadError(
                "DEVICE=cuda but CUDA not available! Set DEVICE=cpu or DEVICE=auto"
            )
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name}")
        return "cuda"

    elif config_device == "cpu":
        logger.info("Using CPU (configured explicitly)")
        return "cpu"

    else:  # "auto"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name} (auto-selected)")
            return "cuda"
        else:
            logger.info("No GPU detected, using CPU (auto-selected)")
            return "cpu"


def get_model_info() -> dict:
    """Get information about loaded model."""
    device = _get_device()

    info = {
        "model_name": settings.MODEL_NAME,
        "device": device,
        "loaded": _model_instance is not None,
        "cache_dir": settings.CACHE_DIR,
    }

    if device == "cuda" and _model_instance is not None:
        allocated = torch.cuda.memory_allocated() / (1024**2)
        info["gpu_memory_mb"] = round(allocated, 2)

    return info


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL LOADER TEST")
    print("=" * 60)

    print("\n🔍 Device detection...")
    device = _get_device()
    print(f"✅ Device: {device}")

    print("\n📥 Loading model (may take 30-60s first time)...")
    try:
        model, processor = get_model()
        print("✅ Model loaded successfully!")

        # Test singleton
        import time

        start = time.time()
        model2, processor2 = get_model()
        elapsed = time.time() - start
        print(f"\n🔄 Second call: {elapsed * 1000:.2f}ms (should be instant)")
        print(f"   Same instance: {model is model2}")

        print("\n📊 Model info:")
        for key, value in get_model_info().items():
            print(f"   {key}: {value}")

        print("\n✅ ALL TESTS PASSED!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
