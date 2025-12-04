"""
FastAPI Service - Image Captioning API

This service provides REST API endpoints for image captioning using BLIP model.

Endpoints:
    POST /api/v1/infer - Generate caption for single image
    POST /api/v1/infer/batch - Generate captions for multiple images
    GET /health - Health check endpoint
"""

import io
import base64
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from PIL import Image

from config import settings
from ml.preprocessing import validate_image, load_and_preprocess, ImageValidationError
from ml.inference import generate_caption, generate_caption_batch, InferenceError
from ml.model_loader import get_model, get_model_info
from utils.exceptions import APIError, InvalidRequestError

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic Models (Request/Response validation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CaptionRequest(BaseModel):
    """Request model for single image caption generation."""

    image: str = Field(
        ...,
        description="Base64 encoded image",
        example="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    )
    max_length: int = Field(
        default=50, ge=10, le=100, description="Maximum caption length"
    )
    num_beams: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Beam search width (1=greedy, 3=balanced, 5+=high quality)",
    )

    @validator("image")
    def validate_base64(cls, v):
        """Validate base64 image string."""
        try:
            base64.b64decode(v)
        except Exception as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        return v


class CaptionResponse(BaseModel):
    """Response model for caption generation."""

    caption: str = Field(..., description="Generated caption")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_name: str = Field(..., description="Model used")
    device: str = Field(..., description="Device used (cpu/cuda)")


class BatchCaptionRequest(BaseModel):
    """Request model for batch caption generation."""

    images: List[str] = Field(
        ..., description="List of base64 encoded images", min_items=1, max_items=10
    )
    max_length: int = Field(default=50, ge=10, le=100)
    num_beams: int = Field(default=3, ge=1, le=10)

    @validator("images")
    def validate_images(cls, v):
        """Validate all images are valid base64."""
        for idx, img in enumerate(v):
            try:
                base64.b64decode(img)
            except Exception as e:
                raise ValueError(f"Image {idx} has invalid base64 encoding: {str(e)}")
        return v


class BatchCaptionResponse(BaseModel):
    """Response model for batch caption generation."""

    results: List[CaptionResponse] = Field(..., description="List of caption results")
    total_processing_time: float = Field(..., description="Total processing time")
    images_processed: int = Field(..., description="Number of images processed")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Error details")
    status_code: int = Field(..., description="HTTP status code")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lifespan Management (Model Preloading)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager - runs on startup and shutdown.

    Startup: Preload ML model
    Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Starting FastAPI service...")
    logger.info(f"Loading model: {settings.MODEL_NAME}")

    try:
        # Preload model (so first request is fast)
        model, processor = get_model()
        logger.info("✅ Model loaded successfully!")
        logger.info(f"Device: {get_model_info()['device']}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        # Continue anyway, will fail on first request

    yield  # Server runs here

    # Shutdown
    logger.info("Shutting down FastAPI service...")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI App
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title="Image Captioning API",
    description="REST API for generating image captions using BLIP model",
    version="1.0.0",
    lifespan=lifespan,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Exception Handlers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.exception_handler(APIError)
async def api_error_handler(request, exc: APIError):
    """Handle custom API errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(ImageValidationError)
async def image_validation_error_handler(request, exc: ImageValidationError):
    """Handle image validation errors."""
    logger.warning(f"Image validation error: {exc}")
    return JSONResponse(
        status_code=400, content={"error": str(exc), "details": {}, "status_code": 400}
    )


@app.exception_handler(InferenceError)
async def inference_error_handler(request, exc: InferenceError):
    """Handle inference errors."""
    logger.error(f"Inference error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Failed to generate caption",
            "details": {"message": str(exc)},
            "status_code": 500,
        },
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and model information.
    """
    model_info = get_model_info()

    return HealthResponse(
        status="healthy", model_loaded=model_info["loaded"], model_info=model_info
    )


@app.post("/api/v1/infer", response_model=CaptionResponse)
async def infer_single(request: CaptionRequest):
    """
    Generate caption for a single image.

    Args:
        request: Caption request with base64 encoded image

    Returns:
        Caption response with generated caption and metadata

    Raises:
        HTTPException: If validation or inference fails
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image)
        logger.info(f"Received image: {len(image_bytes)} bytes")

        # Validate image
        validate_image(image_bytes, max_size=settings.MAX_IMAGE_SIZE)

        # Load and preprocess
        image = load_and_preprocess(image_bytes)
        logger.info(f"Image preprocessed: {image.size}")

        # Generate caption
        result = generate_caption(
            image, max_length=request.max_length, num_beams=request.num_beams
        )

        logger.info(
            f"Caption generated: '{result['caption']}' in {result['processing_time']:.2f}s"
        )

        return CaptionResponse(**result)

    except ImageValidationError as e:
        logger.warning(f"Validation failed: {e}")
        raise
    except InferenceError as e:
        logger.error(f"Inference failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/infer/batch", response_model=BatchCaptionResponse)
async def infer_batch(request: BatchCaptionRequest):
    """
    Generate captions for multiple images.

    Args:
        request: Batch request with list of base64 encoded images

    Returns:
        Batch response with list of captions and metadata

    Raises:
        HTTPException: If validation or inference fails
    """
    import time

    start_time = time.time()

    try:
        # Decode and validate all images
        images = []
        for idx, img_base64 in enumerate(request.images):
            try:
                # Decode
                image_bytes = base64.b64decode(img_base64)

                # Validate
                validate_image(image_bytes, max_size=settings.MAX_IMAGE_SIZE)

                # Load
                image = load_and_preprocess(image_bytes)
                images.append(image)

            except Exception as e:
                raise InvalidRequestError(
                    f"Image {idx} is invalid: {str(e)}", details={"image_index": idx}
                )

        logger.info(f"Batch processing {len(images)} images...")

        # Generate captions (batch processing)
        results = generate_caption_batch(
            images, max_length=request.max_length, num_beams=request.num_beams
        )

        total_time = time.time() - start_time

        logger.info(f"Batch completed: {len(results)} captions in {total_time:.2f}s")

        return BatchCaptionResponse(
            results=[CaptionResponse(**r) for r in results],
            total_processing_time=total_time,
            images_processed=len(results),
        )

    except InvalidRequestError:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch processing failed: {str(e)}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Server (for development)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("Starting Image Captioning API")
    logger.info("=" * 60)
    logger.info(f"Host: {settings.API_HOST}")
    logger.info(f"Port: {settings.API_PORT}")
    logger.info(f"Model: {settings.MODEL_NAME}")
    logger.info("=" * 60)

    uvicorn.run(
        "fastapi_service:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,  # Auto-reload on code changes (dev only)
        log_level=settings.LOG_LEVEL.lower(),
    )
