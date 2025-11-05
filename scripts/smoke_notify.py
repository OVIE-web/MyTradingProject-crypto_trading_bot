# scripts/smoke_notify.py
import asyncio
from src.notifier import TelegramNotifier, send_email_notification

async def main():
    notifier = TelegramNotifier()
    ok = await notifier.send_message("Smoke test: hello from bot")
    print("Telegram ok:", ok)
    ok2 = send_email_notification("Smoke test", "hello from email")
    print("Email ok:", ok2)

if __name__ == "__main__":
    asyncio.run(main())
