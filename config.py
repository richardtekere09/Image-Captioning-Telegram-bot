"""
Configuration Management
========================

Centralized configuration using environment variables and Pydantic settings.
Follows the 12-factor app methodology for configuration.

Usage:
    from config import settings
    print(settings.BOT_TOKEN)
"""

import os
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    
    All settings can be overridden via .env file or environment variables.
    """
    
    # ============================================
    # TELEGRAM BOT CONFIGURATION
    # ============================================
    
    BOT_TOKEN: str = Field(
        ...,
        description="Telegram bot token from @BotFather",
        examples=["1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"]
    )
    
    RATE_LIMIT_SECONDS: float = Field(
        default=2.0,
        ge=0.1,
        le=60.0,
        description="Minimum seconds between requests per user"
    )
    
    MAX_IMAGE_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum image size in bytes"
    )
    
    # ============================================
    # API SERVICE CONFIGURATION
    # ============================================
    
    API_HOST: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    
    API_PORT: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API server port"
    )
    
    API_URL: str = Field(
        default="http://localhost:8000",
        description="API URL for bot to connect to"
    )
    
    # ============================================
    # ML MODEL CONFIGURATION
    # ============================================
    
    MODEL_NAME: str = Field(
        default="Salesforce/blip-image-captioning-base",
        description="Hugging Face model identifier"
    )
    
    DEVICE: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description="Device to run model on: auto (detect), cuda, or cpu"
    )
    
    CACHE_DIR: str = Field(
        default="./models",
        description="Directory to cache downloaded models"
    )
    
    MAX_CAPTION_LENGTH: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum length of generated captions"
    )
    
    # ============================================
    # LOGGING CONFIGURATION
    # ============================================
    
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    LOG_FILE: str = Field(
        default="./logs/app.log",
        description="Path to log file"
    )
    
    # ============================================
    # DEVELOPMENT FLAGS
    # ============================================
    
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    REQUEST_LOGGING: bool = Field(
        default=True,
        description="Log all incoming requests"
    )
    
    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra fields in .env
    )
    
    @field_validator("CACHE_DIR")
    @classmethod
    def create_cache_dir(cls, v: str) -> str:
        """Ensure cache directory exists."""
        os.makedirs(v, exist_ok=True)
        return v
    
    @field_validator("LOG_FILE")
    @classmethod
    def create_log_dir(cls, v: str) -> str:
        """Ensure log directory exists."""
        log_dir = os.path.dirname(v)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        return v
    
    def get_api_endpoint(self, path: str) -> str:
        """
        Get full API endpoint URL.
        
        Args:
            path: API path (e.g., "/infer")
            
        Returns:
            Full URL (e.g., "http://localhost:8000/infer")
        """
        return f"{self.API_URL.rstrip('/')}/{path.lstrip('/')}"


# Global settings instance
# Import this in other modules: from config import settings
settings = Settings()


if __name__ == "__main__":
    """Test configuration loading."""
    print("=" * 60)
    print("CONFIGURATION TEST")
    print("=" * 60)
    
    # Test all settings
    print(f"\n📱 Bot Configuration:")
    print(f"  Token (first 20 chars): {settings.BOT_TOKEN[:20]}...")
    print(f"  Rate Limit: {settings.RATE_LIMIT_SECONDS}s")
    print(f"  Max Image Size: {settings.MAX_IMAGE_SIZE / 1024 / 1024:.1f} MB")
    
    print(f"\n🚀 API Configuration:")
    print(f"  Host: {settings.API_HOST}")
    print(f"  Port: {settings.API_PORT}")
    print(f"  URL: {settings.API_URL}")
    
    print(f"\n🧠 ML Configuration:")
    print(f"  Model: {settings.MODEL_NAME}")
    print(f"  Device: {settings.DEVICE}")
    print(f"  Cache Dir: {settings.CACHE_DIR}")
    print(f"  Max Caption Length: {settings.MAX_CAPTION_LENGTH}")
    
    print(f"\n📝 Logging Configuration:")
    print(f"  Level: {settings.LOG_LEVEL}")
    print(f"  File: {settings.LOG_FILE}")
    
    print(f"\n🔧 Development:")
    print(f"  Debug: {settings.DEBUG}")
    print(f"  Request Logging: {settings.REQUEST_LOGGING}")
    
    print(f"\n✅ Configuration loaded successfully!")
    print("=" * 60)
