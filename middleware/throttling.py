"""
Rate Limiting Middleware
========================

Prevents users from spamming the bot.
"""

import time
import logging
from typing import Dict
from aiogram import BaseMiddleware
from aiogram.types import Message
from config import settings

logger = logging.getLogger(__name__)


class ThrottlingMiddleware(BaseMiddleware):
    """
    Rate limiting middleware.

    Limits how often users can send requests.
    """

    def __init__(self, rate_limit: float = None):
        """
        Initialize throttling middleware.

        Args:
            rate_limit: Minimum seconds between requests (default: from settings)
        """
        super().__init__()
        self.rate_limit = rate_limit or settings.RATE_LIMIT_SECONDS
        self.user_timestamps: Dict[int, float] = {}
        logger.info(f"Throttling enabled: {self.rate_limit}s between requests")

    async def __call__(self, handler, event: Message, data: dict):
        """
        Check rate limit before processing message.

        Args:
            handler: Next handler in chain
            event: Telegram message
            data: Additional data

        Returns:
            Handler result or None if throttled
        """
        user_id = event.from_user.id
        current_time = time.time()

        # Check if user is rate limited
        if user_id in self.user_timestamps:
            last_time = self.user_timestamps[user_id]
            time_passed = current_time - last_time

            if time_passed < self.rate_limit:
                # User is sending too fast
                wait_time = self.rate_limit - time_passed
                logger.warning(f"User {user_id} throttled (wait {wait_time:.1f}s)")

                await event.answer(
                    f"⏱️ Please wait {wait_time:.1f} seconds before sending another image.\n"
                    f"This prevents server overload. Thank you! 😊"
                )
                return  # Don't process the message

        # Update timestamp
        self.user_timestamps[user_id] = current_time

        # Process message
        return await handler(event, data)
