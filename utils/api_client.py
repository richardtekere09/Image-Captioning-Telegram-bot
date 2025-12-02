"""
API Client for FastAPI Service
==============================

Handles HTTP communication with the FastAPI service.
"""

import aiohttp
import base64
import logging
from typing import Dict, Any, Optional
from config import settings

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Raised when API request fails."""

    pass


class APIClient:
    """
    HTTP client for communicating with FastAPI service.

    Uses aiohttp for async requests.
    """

    def __init__(self, base_url: str = None):
        """
        Initialize API client.

        Args:
            base_url: Base URL of API service (default: from settings)
        """
        self.base_url = base_url or settings.API_URL
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(f"API Client initialized: {self.base_url}")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        """Ensure session exists."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if API service is healthy.

        Returns:
            Health status dictionary

        Raises:
            APIClientError: If health check fails
        """
        await self._ensure_session()

        try:
            async with self.session.get(
                f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("API health check: OK")
                    return data
                else:
                    raise APIClientError(f"Health check failed: {response.status}")

        except aiohttp.ClientError as e:
            logger.error(f"Health check failed: {e}")
            raise APIClientError(f"Cannot connect to API: {str(e)}")

    async def generate_caption(
        self, image_bytes: bytes, max_length: int = 50, num_beams: int = 3
    ) -> Dict[str, Any]:
        """
        Generate caption for an image.

        Args:
            image_bytes: Image data as bytes
            max_length: Maximum caption length
            num_beams: Beam search width

        Returns:
            Dictionary with caption and metadata

        Raises:
            APIClientError: If request fails
        """
        await self._ensure_session()

        try:
            # Convert image to base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Prepare request
            request_data = {
                "image": image_base64,
                "max_length": max_length,
                "num_beams": num_beams,
            }

            logger.info(
                f"Sending caption request (image size: {len(image_bytes)} bytes)"
            )

            # Send request
            async with self.session.post(
                f"{self.base_url}/api/v1/infer",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Caption received: '{data['caption']}'")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    raise APIClientError(f"API returned error: {response.status}")

        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise APIClientError(f"Failed to generate caption: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise APIClientError(f"Unexpected error: {str(e)}")

    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("API Client closed")


# Singleton instance
_api_client: Optional[APIClient] = None


def get_api_client() -> APIClient:
    """
    Get singleton API client instance.

    Returns:
        APIClient instance
    """
    global _api_client

    if _api_client is None:
        _api_client = APIClient()

    return _api_client


# Test
if __name__ == "__main__":
    import asyncio

    async def test_client():
        """Test API client."""
        print("=" * 60)
        print("API CLIENT TEST")
        print("=" * 60)

        client = get_api_client()

        # Test 1: Health check
        print("\n🏥 Testing health check...")
        try:
            health = await client.health_check()
            print(f"✅ API is healthy!")
            print(f"   Status: {health['status']}")
            print(f"   Model loaded: {health['model_loaded']}")
        except APIClientError as e:
            print(f"❌ Health check failed: {e}")
            return

        # Test 2: Caption generation (would need real image)
        print("\n📝 Caption generation test skipped (needs real image)")
        print("   Use in telegram_bot.py for real testing")

        # Close
        await client.close()

        print("\n✅ API Client tests complete!")

    asyncio.run(test_client())
