# scripts/smoke_notify.py
import asyncio
from src.notifier import TelegramNotifier, send_email_notification

async def main():
    notifier = TelegramNotifier()
    ok = await notifier.send_message("Smoke test: Crypto Trading Bot")
    print("Telegram ok:", ok)
    ok2 = send_email_notification("Smoke test", "Hello From Crypto Trading Bot")
    print("Email ok:", ok2)

if __name__ == "__main__":
    asyncio.run(main())
