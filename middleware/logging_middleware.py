"""
Logging Middleware
==================

Logs all incoming messages for monitoring and debugging.
"""

import logging
from aiogram import BaseMiddleware
from aiogram.types import Message

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseMiddleware):
    """
    Logs all incoming messages.
    """

    async def __call__(self, handler, event: Message, data: dict):
        """
        Log message details.

        Args:
            handler: Next handler in chain
            event: Telegram message
            data: Additional data

        Returns:
            Handler result
        """
        user = event.from_user

        # Determine message type
        if event.photo:
            msg_type = "photo"
        elif event.text:
            msg_type = f"text: {event.text[:50]}"
        else:
            msg_type = "other"

        logger.info(f"Message from user {user.id} (@{user.username}): {msg_type}")

        # Process message
        result = await handler(event, data)

        return result
