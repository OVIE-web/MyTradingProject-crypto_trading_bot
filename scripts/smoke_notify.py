"""Script to test notification."""

import asyncio
import logging

from src.notifications.notifier import TelegramNotifier, send_email_notification

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    """Run notification smoke test."""
    logger.info("Starting notification smoke test...")

    # 1. Test Telegram
    notifier = TelegramNotifier()
    if notifier.enabled:
        ok = await notifier.send_message("Smoke test: Hello From Trading Bot")
        logger.info("Telegram notification sent: %s", ok)
    else:
        logger.warning("Telegram notifier is disabled (check environment variables).")

    # 2. Test Email
    ok2 = send_email_notification("Smoke test", "Hello From Trading Bot")
    logger.info("Email notification sent: %s", ok2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.exception("Fatal error during smoke test: %s", e)
