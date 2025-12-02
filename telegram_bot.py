"""
Telegram Bot - Image Captioning
================================

Main bot file that handles user interactions.

Commands:
    /start - Welcome message
    /help  - Show help

Features:
    - Photo captioning
    - Rate limiting
    - Error handling
"""

import asyncio
import logging
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, BufferedInputFile
from aiogram.enums import ParseMode

from config import settings
from utils.api_client import get_api_client, APIClientError
from middleware.throttling import ThrottlingMiddleware
from middleware.logging_middleware import LoggingMiddleware
from middleware.error_handler import ErrorHandlerMiddleware

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(settings.LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize bot and dispatcher
bot = Bot(token=settings.BOT_TOKEN)
dp = Dispatcher()

# Register middleware (order matters!)
dp.message.middleware(LoggingMiddleware())
dp.message.middleware(ErrorHandlerMiddleware())
dp.message.middleware(ThrottlingMiddleware())

# API client
api_client = get_api_client()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command Handlers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command."""
    await message.answer(
        "👋 **Welcome to Image Captioning Bot!**\n\n"
        "I can describe your images using AI! 🤖\n\n"
        "**How to use:**\n"
        "1. Send me a photo 📸\n"
        "2. Wait a few seconds ⏱️\n"
        "3. Get AI-generated caption! ✨\n\n"
        "**Commands:**\n"
        "/start - Show this message\n"
        "/help - Get help\n\n"
        "**Try it now! Send me a photo!** 🎉",
        parse_mode=ParseMode.MARKDOWN,
    )
    logger.info(f"User {message.from_user.id} started the bot")


@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Handle /help command."""
    await message.answer(
        "📖 **Help - Image Captioning Bot**\n\n"
        "**What I do:**\n"
        "I analyze your photos and generate descriptive captions using "
        "advanced AI (BLIP model).\n\n"
        "**How to use:**\n"
        "• Send me any photo\n"
        "• I'll analyze it and tell you what I see\n"
        "• Works with: cats, dogs, people, objects, scenes, etc.\n\n"
        "**Tips:**\n"
        "• Clear photos work best\n"
        "• Max file size: 10 MB\n"
        "• Supported formats: JPEG, PNG, WEBP\n"
        "• Wait 2 seconds between photos (prevents spam)\n\n"
        "**Processing time:**\n"
        "• Usually 2-5 seconds per photo\n"
        "• First request may take longer (model loading)\n\n"
        "**Privacy:**\n"
        "• Images are not stored\n"
        "• Processed in real-time only\n"
        "• Your data is safe! 🔒\n\n"
        "**Questions?** Just send /help again!",
        parse_mode=ParseMode.MARKDOWN,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Photo Handler (Main Feature!)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dp.message(F.photo)
async def handle_photo(message: Message):
    """
    Handle photo messages - main feature!

    Flow:
    1. User sends photo
    2. Download photo from Telegram
    3. Send to FastAPI service
    4. Get caption back
    5. Send caption to user
    """
    logger.info(f"Processing photo from user {message.from_user.id}")

    # Send "processing" message
    processing_msg = await message.answer(
        "🤖 Analyzing your image...\n⏳ Please wait..."
    )

    try:
        # Get largest photo (best quality)
        photo = message.photo[-1]

        # Check file size
        if photo.file_size > settings.MAX_IMAGE_SIZE:
            await processing_msg.edit_text(
                f"❌ Image too large!\n\n"
                f"Your image: {photo.file_size / 1024 / 1024:.1f} MB\n"
                f"Max allowed: {settings.MAX_IMAGE_SIZE / 1024 / 1024:.1f} MB\n\n"
                f"Please send a smaller image."
            )
            return

        # Download photo
        logger.info(f"Downloading photo: {photo.file_id}")
        file = await bot.get_file(photo.file_id)
        image_bytes = await bot.download_file(file.file_path)

        # Read bytes
        image_data = image_bytes.read()
        logger.info(f"Downloaded {len(image_data)} bytes")

        # Send to API
        logger.info("Sending to API for caption generation...")
        result = await api_client.generate_caption(
            image_bytes=image_data, max_length=settings.MAX_CAPTION_LENGTH, num_beams=3
        )

        # Extract results
        caption = result["caption"]
        processing_time = result["processing_time"]
        device = result["device"]

        logger.info(f"Caption generated: '{caption}' ({processing_time:.2f}s)")

        # Send result to user
        await processing_msg.edit_text(
            f"✨ **Caption:**\n{caption}\n\n"
            f"⏱️ Processing time: {processing_time:.2f}s\n"
            f"💻 Device: {device.upper()}\n"
            f"🧠 Model: BLIP",
            parse_mode=ParseMode.MARKDOWN,
        )

        logger.info(f"Successfully processed photo for user {message.from_user.id}")

    except APIClientError as e:
        logger.error(f"API error: {e}")
        await processing_msg.edit_text(
            "❌ **API Error**\n\n"
            "The caption service is temporarily unavailable.\n"
            "Please try again in a moment.\n\n"
            f"Error: {str(e)[:100]}"
        )

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        await processing_msg.edit_text(
            "❌ **Error**\n\n"
            "Something went wrong while processing your image.\n"
            "Please try again.\n\n"
            f"Error: {str(e)[:100]}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Text Handler (Fallback)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dp.message(F.text)
async def handle_text(message: Message):
    """Handle text messages (not commands)."""
    await message.answer(
        "📸 **Please send me a photo!**\n\n"
        "I can only process images, not text.\n\n"
        "Send me any photo and I'll describe what I see! ✨\n\n"
        "Need help? Send /help"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Startup/Shutdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def on_startup():
    """Run on bot startup."""
    logger.info("=" * 60)
    logger.info("TELEGRAM BOT STARTING")
    logger.info("=" * 60)

    # Check API health
    try:
        logger.info("Checking API health...")
        health = await api_client.health_check()
        logger.info(f"✅ API is healthy!")
        logger.info(f"   Model: {health['model_info']['model_name']}")
        logger.info(f"   Device: {health['model_info']['device']}")
    except APIClientError as e:
        logger.error(f"❌ API health check failed: {e}")
        logger.error("   Bot will start anyway, but captioning won't work!")

    logger.info("=" * 60)
    logger.info("✅ Bot is ready!")
    logger.info("=" * 60)


async def on_shutdown():
    """Run on bot shutdown."""
    logger.info("Shutting down bot...")
    await api_client.close()
    await bot.session.close()
    logger.info("Bot stopped.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def main():
    """Main function."""
    # Run startup
    await on_startup()

    try:
        # Start polling
        await dp.start_polling(bot)
    finally:
        # Run shutdown
        await on_shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
