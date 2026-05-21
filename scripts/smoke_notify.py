"""Smoke-test configured notification channels.

Run with:
    python -m scripts.smoke_notify
    python -m scripts.smoke_notify --skip-email
    python -m scripts.smoke_notify --skip-telegram
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.env_loader import load_environment  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.services.notifications import (  # noqa: E402
    TelegramNotifier,
    send_email_notification,
)

DEFAULT_SUBJECT = "OvieX Quant Engine notification smoke test"
DEFAULT_MESSAGE = "Smoke test: notification channel is configured correctly."

load_environment()
setup_logging()
logger = get_logger(__name__)


async def run_smoke_test(
    *,
    subject: str = DEFAULT_SUBJECT,
    message: str = DEFAULT_MESSAGE,
    skip_telegram: bool = False,
    skip_email: bool = False,
) -> bool:
    """Send a smoke-test message through configured notification channels."""
    logger.info("Starting notification smoke test.")
    results: list[bool] = []

    if not skip_telegram:
        notifier = TelegramNotifier()
        if notifier.enabled:
            telegram_sent = await notifier.send_message(message)
            logger.info("Telegram smoke test result: %s", telegram_sent)
            results.append(telegram_sent)
        else:
            logger.warning(
                "Telegram notifier is disabled. Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
            )
            results.append(False)

    if not skip_email:
        email_sent = send_email_notification(subject, message)
        logger.info("Email smoke test result: %s", email_sent)
        results.append(email_sent)

    if not results:
        logger.warning("No notification channels were selected.")
        return False

    succeeded = any(results)
    logger.info("Notification smoke test completed. success=%s", succeeded)
    return succeeded


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Smoke-test notification channels.")
    parser.add_argument(
        "--subject",
        default=DEFAULT_SUBJECT,
        help="Email subject for the smoke test.",
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_MESSAGE,
        help="Message body sent to enabled notification channels.",
    )
    parser.add_argument(
        "--skip-telegram",
        action="store_true",
        help="Skip the Telegram smoke test.",
    )
    parser.add_argument(
        "--skip-email",
        action="store_true",
        help="Skip the email smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the notification smoke test and return a process exit code."""
    args = parse_args()

    try:
        succeeded = asyncio.run(
            run_smoke_test(
                subject=args.subject,
                message=args.message,
                skip_telegram=args.skip_telegram,
                skip_email=args.skip_email,
            )
        )
    except KeyboardInterrupt:
        logger.info("Notification smoke test interrupted by user.")
        return 130
    except Exception as exc:
        logger.exception("Fatal error during notification smoke test: %s", exc)
        return 1

    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
