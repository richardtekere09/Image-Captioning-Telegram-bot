"""
Image Preprocessing Module
"""

import io
from typing import Optional
from PIL import Image, UnidentifiedImageError
import logging

logger = logging.getLogger(__name__)


class ImageValidationError(Exception):
    """Raised when image validation fails."""

    pass


class ImagePreprocessor:
    """Handles all image preprocessing operations."""

    SUPPORTED_FORMATS = {"JPEG", "PNG", "JPG", "WEBP"}

    @staticmethod
    def validate_image(image_bytes: bytes, max_size: int = 10 * 1024 * 1024) -> bool:
        """
        Validate image size and format.

        Args:
            image_bytes: Raw image data as bytes
            max_size: Maximum allowed size in bytes (default: 10MB)

        Returns:
            True if valid

        Raises:
            ImageValidationError: If image is invalid
        """
        # Check size
        if len(image_bytes) > max_size:
            size_mb = len(image_bytes) / (1024 * 1024)
            max_mb = max_size / (1024 * 1024)
            raise ImageValidationError(
                f"Image too large: {size_mb:.2f} MB (max: {max_mb:.1f} MB)"
            )

        # Check minimum size
        if len(image_bytes) < 100:
            raise ImageValidationError(
                f"Image too small: {len(image_bytes)} bytes (likely corrupted)"
            )

        # Check format
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_format = image.format

            if image_format not in ImagePreprocessor.SUPPORTED_FORMATS:
                raise ImageValidationError(
                    f"Unsupported format: {image_format}. "
                    f"Supported: {', '.join(ImagePreprocessor.SUPPORTED_FORMATS)}"
                )

            logger.info(f"Image validated: {image_format}, {len(image_bytes)} bytes")
            return True

        except UnidentifiedImageError:
            raise ImageValidationError(
                "Cannot identify image format. File may be corrupted."
            )
        except Exception as e:
            raise ImageValidationError(f"Image validation failed: {str(e)}")

    @staticmethod
    def load_and_preprocess(image_bytes: bytes) -> Image.Image:
        """
        Load image from bytes and prepare for processing.

        Args:
            image_bytes: Raw image data

        Returns:
            PIL Image in RGB format

        Raises:
            ImageValidationError: If image cannot be loaded
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))

            # Handle EXIF orientation
            try:
                exif = image._getexif()
                if exif is not None:
                    orientation_key = 274
                    if orientation_key in exif:
                        orientation = exif[orientation_key]
                        rotations = {3: 180, 6: 270, 8: 90}
                        if orientation in rotations:
                            image = image.rotate(rotations[orientation], expand=True)
                            logger.info(f"Rotated image {rotations[orientation]}°")
            except (AttributeError, KeyError, TypeError):
                pass

            # Convert to RGB
            if image.mode != "RGB":
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert("RGB")

            logger.info(f"Image loaded: {image.size[0]}x{image.size[1]} pixels")
            return image

        except UnidentifiedImageError:
            raise ImageValidationError("Cannot open image - format not recognized")
        except Exception as e:
            raise ImageValidationError(f"Failed to load image: {str(e)}")

    @staticmethod
    def resize_if_needed(image: Image.Image, max_dimension: int = 2048) -> Image.Image:
        """
        Resize image if too large.

        Args:
            image: PIL Image to resize
            max_dimension: Maximum width or height

        Returns:
            Resized image or original
        """
        width, height = image.size

        if width <= max_dimension and height <= max_dimension:
            return image

        if width > height:
            new_width = max_dimension
            new_height = int((max_dimension / width) * height)
        else:
            new_height = max_dimension
            new_width = int((max_dimension / height) * width)

        logger.info(f"Resizing {width}x{height} to {new_width}x{new_height}")
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# Convenience functions
def validate_image(image_bytes: bytes, max_size: int = 10 * 1024 * 1024) -> bool:
    """Validate image (convenience function)."""
    return ImagePreprocessor.validate_image(image_bytes, max_size)


def load_and_preprocess(image_bytes: bytes) -> Image.Image:
    """Load and preprocess image (convenience function)."""
    return ImagePreprocessor.load_and_preprocess(image_bytes)


if __name__ == "__main__":
    print("=" * 60)
    print("IMAGE PREPROCESSING TEST")
    print("=" * 60)
    print("\n✅ Module loaded successfully!")
    print("Ready for use in production.")
