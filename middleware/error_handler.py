"""
Error Handler Middleware
========================

Catches and handles errors gracefully.
"""

import logging
from aiogram import BaseMiddleware
from aiogram.types import Message, ErrorEvent

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseMiddleware):
    """
    Global error handler.
    """

    async def __call__(self, handler, event: Message, data: dict):
        """
        Catch and handle errors.

        Args:
            handler: Next handler in chain
            event: Telegram message
            data: Additional data

        Returns:
            Handler result
        """
        try:
            return await handler(event, data)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

            await event.answer(
                "❌ Sorry, something went wrong! 😔\n\n"
                "Please try again in a moment. If the problem persists, "
                "contact support.\n\n"
                f"Error: {str(e)[:100]}"
            )

            return None
