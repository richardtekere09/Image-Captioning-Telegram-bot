"""
Inference Module - Caption Generation
"""

import time
import torch
from PIL import Image
from typing import Dict, List, Union
import logging
from config import settings
from ml.model_loader import get_model

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    """Raised when caption generation fails."""

    pass


def generate_caption(
    image: Image.Image,
    max_length: int = None,
    num_beams: int = 3,
    temperature: float = 1.0,
) -> Dict[str, Union[str, float]]:
    """
    Generate caption for a single image.

    Args:
        image: PIL Image to caption
        max_length: Maximum caption length
        num_beams: Beam search width (1=greedy, 3=balanced, 5=best)
        temperature: Randomness (1.0=default)

    Returns:
        Dictionary with caption, processing_time, model_name, device

    Raises:
        InferenceError: If generation fails
    """
    start_time = time.time()

    try:
        model, processor = get_model()
        device = next(model.parameters()).device.type

        if max_length is None:
            max_length = settings.MAX_CAPTION_LENGTH

        logger.debug(f"Generating caption (max_length={max_length}, beams={num_beams})")

        # Preprocess image
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Generate caption
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True,
            )

        # Decode to text
        caption = processor.decode(output_ids[0], skip_special_tokens=True)

        # Post-process
        caption = _post_process_caption(caption)

        processing_time = time.time() - start_time

        logger.info(f"Caption generated in {processing_time:.2f}s: '{caption}'")

        return {
            "caption": caption,
            "processing_time": processing_time,
            "model_name": settings.MODEL_NAME,
            "device": device,
        }

    except Exception as e:
        error_msg = f"Caption generation failed: {str(e)}"
        logger.error(error_msg)
        raise InferenceError(error_msg) from e


def generate_caption_batch(
    images: List[Image.Image], max_length: int = None, num_beams: int = 3
) -> List[Dict[str, Union[str, float]]]:
    """
    Generate captions for multiple images (batch processing).

    Args:
        images: List of PIL Images
        max_length: Maximum caption length
        num_beams: Beam search width

    Returns:
        List of result dictionaries
    """
    start_time = time.time()

    try:
        model, processor = get_model()
        device = next(model.parameters()).device.type

        if max_length is None:
            max_length = settings.MAX_CAPTION_LENGTH

        logger.info(f"Batch processing {len(images)} images...")

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        captions = processor.batch_decode(output_ids, skip_special_tokens=True)

        total_time = time.time() - start_time
        time_per_image = total_time / len(images)

        results = []
        for i, caption in enumerate(captions):
            caption = _post_process_caption(caption)
            results.append(
                {
                    "caption": caption,
                    "processing_time": time_per_image,
                    "model_name": settings.MODEL_NAME,
                    "device": device,
                    "batch_index": i,
                }
            )

        logger.info(f"Batch: {total_time:.2f}s total ({time_per_image:.2f}s/image)")

        return results

    except Exception as e:
        error_msg = f"Batch generation failed: {str(e)}"
        logger.error(error_msg)
        raise InferenceError(error_msg) from e


def _post_process_caption(caption: str) -> str:
    """Clean and format caption."""
    caption = " ".join(caption.split())
    caption = caption.strip()

    if caption:
        caption = caption[0].upper() + caption[1:]

    if caption and caption[-1] not in [".", "!", "?"]:
        caption += "."

    return caption

if __name__ == "__main__":
    """Test caption generation with real images."""
    import sys
    import os

    print("=" * 60)
    print("INFERENCE MODULE TEST")
    print("=" * 60)

    # Check if cat image exists
    cat_image_path = "data/sample_images/cat_demo.jpeg"

    if os.path.exists(cat_image_path):
        # Test with real cat image
        print(f"\n📸 Found cat image: {cat_image_path}")
        print("Loading image...")

        try:
            image = Image.open(cat_image_path)
            print(
                f"✅ Image loaded: {image.size[0]}x{image.size[1]} pixels, {image.mode} mode"
            )

            # Test 1: Basic caption generation
            print("\n" + "=" * 60)
            print("TEST 1: Basic Caption Generation")
            print("=" * 60)
            print("🤖 Generating caption...")
            print("(First time: loads model ~30-60s, please wait...)")

            result = generate_caption(image)

            print("\n✅ Caption generated successfully!")
            print(f"\n   📝 Caption: '{result['caption']}'")
            print(f"   ⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"   🧠 Model: {result['model_name']}")
            print(f"   💻 Device: {result['device']}")

            # Test 2: Different num_beams
            print("\n" + "=" * 60)
            print("TEST 2: Different num_beams (Quality vs Speed)")
            print("=" * 60)

            for num_beams in [1, 3, 5]:
                print(f"\n🔬 Testing num_beams={num_beams}...")
                result = generate_caption(image, num_beams=num_beams)
                print(f"   Caption: '{result['caption']}'")
                print(f"   Time: {result['processing_time']:.2f}s")

            # Test 3: Different max_length
            print("\n" + "=" * 60)
            print("TEST 3: Different max_length (Caption Length)")
            print("=" * 60)

            for max_len in [10, 30, 50]:
                print(f"\n📏 Testing max_length={max_len}...")
                result = generate_caption(image, max_length=max_len)
                print(f"   Caption: '{result['caption']}'")

            print("\n" + "=" * 60)
            print("✅ ALL TESTS PASSED WITH REAL IMAGE!")
            print("=" * 60)
            print("\n🎉 Inference module is production-ready!")

        except Exception as e:
            print(f"\n❌ Error testing with cat image: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    else:
        # Fallback: Test with synthetic image
        print(f"\n⚠️  Cat image not found at: {cat_image_path}")
        print("Using synthetic test image instead...")

        print("\n🎨 Creating test image (red square)...")
        test_image = Image.new("RGB", (500, 500), color="red")
        print("✅ Test image created")

        print("\n📝 Generating caption...")
        try:
            result = generate_caption(test_image)
            print("✅ Caption generated!")
            print(f"\n   Caption: '{result['caption']}'")
            print(f"   Time: {result['processing_time']:.2f}s")
            print(f"   Device: {result['device']}")

            print("\n✅ BASIC TESTS PASSED!")
            print("\n💡 To test with real image:")
            print(f"   1. Place image at: {cat_image_path}")
            print(f"   2. Run: python -m ml.inference")

        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)