import asyncio
import os
import time
from collections import deque

from aiohttp import web, ClientSession
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

API_ID = int(os.environ.get("API_ID", "0"))
API_HASH = os.environ.get("API_HASH", "").strip()
SESSION_STRING = os.environ.get("SESSION_STRING", "").strip()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

PORT = int(os.environ.get("PORT", "8080"))
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "").strip()


# ============================================================
# VALIDATION
# ============================================================

if not API_ID:
    raise RuntimeError("❌ API_ID не установлен")

if not API_HASH:
    raise RuntimeError("❌ API_HASH не установлен")

if not SESSION_STRING:
    raise RuntimeError("❌ SESSION_STRING не установлен")


# ============================================================
# TELEGRAM CLIENT
# ============================================================

client = TelegramClient(
    StringSession(SESSION_STRING),
    API_ID,
    API_HASH,

    # Автоматическое восстановление соединения
    auto_reconnect=True,
    connection_retries=10,
    retry_delay=5,
)


# ============================================================
# GEMINI
# ============================================================

ai_client = None

if GEMINI_API_KEY:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# BOT STATE
# ============================================================

bot_state = {
    "afk": False,
    "afk_reason": "",
    "afk_messages": [],

    "mode": None,

    "spam_active": False,
    "spam_task": None,
}


# Последние сообщения для отслеживания удалений
msg_cache = deque(maxlen=500)


# ============================================================
# WEB SERVER FOR RENDER
# ============================================================

async def handle_root(request):
    return web.Response(
        text="Telegram Userbot is running.",
        content_type="text/plain"
    )


async def handle_health(request):
    connected = client.is_connected()

    return web.json_response({
        "status": "ok",
        "telegram_connected": connected,
        "gemini_configured": ai_client is not None,
    })


async def start_web_server():
    app = web.Application()

    app.router.add_get("/", handle_root)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(
        runner,
        host="0.0.0.0",
        port=PORT
    )

    await site.start()

    print(f"🌐 Web server запущен на порту {PORT}")

    return runner


# ============================================================
# OPTIONAL SELF-PING
# ============================================================

async def self_ping_loop():
    """
    Дополнительный self-ping.

    ВАЖНО:
    На бесплатном Render это НЕ является гарантией
    постоянной работы сервиса.
    """

    if not RENDER_EXTERNAL_URL:
        print("ℹ️ RENDER_EXTERNAL_URL не установлен — self-ping отключён")
        return

    print(f"🔄 Self-ping включён: {RENDER_EXTERNAL_URL}")

    timeout = 20

    async with ClientSession() as session:
        while True:
            try:
                await asyncio.sleep(10 * 60)

                async with session.get(
                    RENDER_EXTERNAL_URL,
                    timeout=timeout
                ) as response:

                    print(
                        f"🔄 Self-ping: HTTP {response.status}"
                    )

            except asyncio.CancelledError:
                break

            except Exception as e:
                print(
                    f"⚠️ Self-ping ошибка: "
                    f"{type(e).__name__}: {e}"
                )


# ============================================================
# GEMINI
# ============================================================

async def fast_gemini_generate(
    prompt: str,
    system_instruction: str | None = None
) -> str:

    if not ai_client:
        return "❌ GEMINI_API_KEY не настроен."

    try:
        config = types.GenerateContentConfig(
            system_instruction=(
                system_instruction
                or "Отвечай максимально кратко, понятно и по существу."
            ),
            max_output_tokens=350,
            temperature=0.5,
        )

        response = await ai_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )

        return response.text or "❌ Gemini вернул пустой ответ."

    except Exception as e:
        print(
            f"❌ Gemini error: "
            f"{type(e).__name__}: {e}"
        )

        return f"❌ Ошибка Gemini: {e}"


# ============================================================
# DEBUG / MESSAGE CACHE
# ============================================================

@client.on(events.NewMessage)
async def message_debug_and_cache(event):

    try:
        # DEBUG LOG
        print(
            f"📩 MESSAGE | "
            f"chat={event.chat_id} | "
            f"outgoing={event.out} | "
            f"text={repr(event.raw_text[:100])}"
        )

        # CACHE
        if event.message and event.message.text:
            msg_cache.append(event.message)

    except Exception as e:
        print(
            f"⚠️ Message listener error: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# DELETED MESSAGE LOGGER
# ============================================================

@client.on(events.MessageDeleted)
async def deleted_logger(event):

    try:

        for deleted_id in event.deleted_ids:

            for cached in list(msg_cache):

                if cached.id != deleted_id:
                    continue

                try:
                    sender = await cached.get_sender()

                    sender_name = (
                        getattr(sender, "first_name", None)
                        or getattr(sender, "username", None)
                        or "Неизвестно"
                    )

                    text = cached.text or "[без текста]"

                    log_text = (
                        "🗑 **Удалено сообщение!**\n\n"
                        f"👤 **От:** {sender_name}\n"
                        f"🆔 `{cached.sender_id}`\n"
                        f"💬 **Текст:** {text}"
                    )

                    await client.send_message(
                        "me",
                        log_text
                    )

                except Exception as e:
                    print(
                        f"⚠️ Ошибка логирования удаления: "
                        f"{type(e).__name__}: {e}"
                    )

                break

    except Exception as e:
        print(
            f"⚠️ MessageDeleted error: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# PING
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.ping$"
    )
)
async def ping_handler(event):

    print("🔥 PING HANDLER")

    start = time.perf_counter()

    try:
        await event.edit("🏓 Pinging...")

        elapsed = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        await event.edit(
            f"🚀 **Pong!**\n"
            f"⚡ Задержка: `{elapsed} ms`"
        )

    except Exception as e:
        print(
            f"❌ Ping error: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# HELP
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.help$"
    )
)
async def help_handler(event):

    print("🔥 HELP HANDLER")

    help_text = (
        "⚙️ **Юзербот — команды**\n\n"

        "🏓 `.ping` — проверка работы\n"
        "📚 `.help` — список команд\n\n"

        "💤 `.afk [причина]` — включить AFK\n"
        "🌅 `.unafk` — выключить AFK\n\n"

        "🧠 `.ai [текст]` — Gemini\n"
        "📝 `.remember [кол-во]` — саммари чата\n"
        "🎭 `.clone` — копирование стиля\n\n"

        "😺 `.catgirl` — режим catgirl\n"
        "😤 `.tsundere` — режим tsundere\n"
        "🔄 `.reset` — сброс режима\n\n"

        "📨 `.spam [кол-во] [текст]`\n"
        "⏹ `.stop` — остановить spam"
    )

    try:
        await event.edit(help_text)

    except Exception as e:
        print(
            f"❌ Help error: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# AFK ON
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.afk(?:\s+(.+))?$"
    )
)
async def afk_on(event):

    reason = (
        event.pattern_match.group(1)
        or "Занят"
    )

    bot_state["afk"] = True
    bot_state["afk_reason"] = reason
    bot_state["afk_messages"] = []

    print(f"💤 AFK ON: {reason}")

    await event.edit(
        f"💤 **Режим AFK включен.**\n"
        f"Причина: `{reason}`"
    )


# ============================================================
# AFK OFF
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.unafk$"
    )
)
async def afk_off(event):

    if not bot_state["afk"]:
        await event.edit(
            "❌ Вы не находитесь в AFK."
        )
        return

    count = len(
        bot_state["afk_messages"]
    )

    bot_state["afk"] = False
    bot_state["afk_reason"] = ""

    await event.edit(
        "🌅 **AFK выключен.**\n\n"
        f"📨 Пока вас не было, "
        f"вам написали: `{count}` раз(а)."
    )


# ============================================================
# AFK LISTENER
# ============================================================

@client.on(events.NewMessage(incoming=True))
async def afk_listener(event):

    if not bot_state["afk"]:
        return

    if not (event.is_private or event.mentioned):
        return

    try:

        sender = await event.get_sender()

        name = (
            getattr(sender, "first_name", None)
            or getattr(sender, "username", None)
            or "Кто-то"
        )

        text = event.raw_text or "[без текста]"

        bot_state["afk_messages"].append(
            f"{name}: {text}"
        )

        await event.reply(
            "🤖 **Я сейчас AFK.**\n"
            f"Причина: `{bot_state['afk_reason']}`"
        )

    except Exception as e:
        print(
            f"❌ AFK error: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# GEMINI AI
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.ai\s+(.+)"
    )
)
async def ai_handler(event):

    prompt = event.pattern_match.group(1)

    print(f"🧠 AI request: {prompt[:100]}")

    if not ai_client:
        await event.edit(
            "❌ `GEMINI_API_KEY` не настроен."
        )
        return

    try:

        await event.edit("🧠 *Генерация...*")

        response = await ai_client.aio.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=400,
                temperature=0.6,
            )
        )

        full_text = ""
        last_update = time.time()

        async for chunk in response:

            if not chunk.text:
                continue

            full_text += chunk.text

            # Не редактируем слишком часто
            if (
                time.time() - last_update >= 1.5
                and full_text
            ):
                try:
                    await event.edit(
                        full_text + " ▌"
                    )
                    last_update = time.time()
                except Exception:
                    pass

        if not full_text:
            full_text = "❌ Gemini не вернул текст."

        await event.edit(full_text)

    except Exception as e:

        print(
            f"❌ AI error: "
            f"{type(e).__name__}: {e}"
        )

        await event.edit(
            f"❌ Ошибка Gemini:\n`{str(e)[:500]}`"
        )


# ============================================================
# REMEMBER
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.remember(?:\s+(\d+))?$"
    )
)
async def remember_handler(event):

    try:

        limit = int(
            event.pattern_match.group(1)
            or 25
        )

        # Защита от слишком больших запросов
        limit = max(1, min(limit, 100))

        await event.edit(
            f"📊 Анализирую последние `{limit}` сообщений..."
        )

        raw_messages = await client.get_messages(
            event.chat_id,
            limit=limit
        )

        history = []

        for message in reversed(raw_messages):

            if not message.text:
                continue

            try:
                sender = await message.get_sender()

                name = (
                    getattr(sender, "first_name", None)
                    or getattr(sender, "username", None)
                    or "User"
                )

            except Exception:
                name = "User"

            history.append(
                f"{name}: {message.text}"
            )

        if not history:
            await event.edit(
                "❌ Не удалось найти текстовые сообщения."
            )
            return

        context_text = "\n".join(history)

        prompt = (
            "Сделай краткое и понятное саммари "
            "этой переписки на русском языке.\n\n"
            f"{context_text}"
        )

        result = await fast_gemini_generate(prompt)

        await event.edit(
            f"📝 **Краткая выжимка:**\n\n{result}"
        )

    except Exception as e:

        print(
            f"❌ Remember error: "
            f"{type(e).__name__}: {e}"
        )

        await event.edit(
            f"❌ Ошибка: `{str(e)[:500]}`"
        )


# ============================================================
# CLONE
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.clone$"
    )
)
async def clone_handler(event):

    try:

        reply = await event.get_reply_message()

        if not reply or not reply.sender_id:
            await event.edit(
                "❌ Используй `.clone` ответом "
                "на сообщение пользователя."
            )
            return

        await event.edit(
            "🎭 Анализирую манеру речи..."
        )

        messages = await client.get_messages(
            event.chat_id,
            limit=40,
            from_user=reply.sender_id
        )

        texts = [
            m.text
            for m in messages
            if m.text
        ]

        if not texts:
            await event.edit(
                "❌ Недостаточно сообщений "
                "для анализа."
            )
            return

        sample = "\n".join(
            texts[:15]
        )

        prompt = (
            "Проанализируй стиль автора "
            "по примерам ниже и напиши одно "
            "короткое предложение в похожем стиле. "
            "Сохрани характерную пунктуацию, "
            "сленг и манеру речи.\n\n"
            f"{sample}"
        )

        result = await fast_gemini_generate(prompt)

        await event.edit(result)

    except Exception as e:

        print(
            f"❌ Clone error: "
            f"{type(e).__name__}: {e}"
        )

        await event.edit(
            f"❌ Ошибка: `{str(e)[:500]}`"
        )


# ============================================================
# PERSONALITY MODES
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.(catgirl|tsundere|reset)$"
    )
)
async def mode_handler(event):

    command = event.pattern_match.group(1)

    if command == "reset":

        bot_state["mode"] = None

        await event.edit(
            "🔄 Режим личности сброшен."
        )

        return

    bot_state["mode"] = command

    await event.edit(
        f"🎭 Режим **{command}** активирован!"
    )


# ============================================================
# SPAM
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.spam\s+(\d+)\s+(.+)"
    )
)
async def spam_handler(event):

    try:

        count = int(
            event.pattern_match.group(1)
        )

        text = event.pattern_match.group(2)

        # Разумный предел
        count = max(1, min(count, 100))

        # Останавливаем старую задачу
        if bot_state["spam_task"]:

            bot_state["spam_active"] = False

            try:
                await bot_state["spam_task"]
            except Exception:
                pass

        await event.delete()

        bot_state["spam_active"] = True

        async def spam_loop():

            try:

                for i in range(count):

                    if not bot_state["spam_active"]:
                        break

                    await client.send_message(
                        event.chat_id,
                        text
                    )

                    await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                pass

            except Exception as e:

                print(
                    f"❌ Spam error: "
                    f"{type(e).__name__}: {e}"
                )

            finally:

                bot_state["spam_active"] = False
                bot_state["spam_task"] = None

        task = asyncio.create_task(
            spam_loop()
        )

        bot_state["spam_task"] = task

    except Exception as e:

        print(
            f"❌ Spam handler error: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# STOP SPAM
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.stop$"
    )
)
async def stop_spam(event):

    bot_state["spam_active"] = False

    task = bot_state.get("spam_task")

    if task and not task.done():
        task.cancel()

    bot_state["spam_task"] = None

    await event.edit(
        "⏹ **Процесс остановлен.**"
    )


# ============================================================
# TELEGRAM CONNECTION LOGGING
# ============================================================

@client.on(events.Raw)
async def raw_update_debug(event):
    # Специально ничего не выводим,
    # чтобы не засорять Render Logs.
    pass


# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 60)
    print("🚀 ЗАПУСК TELEGRAM USERBOT")
    print("=" * 60)

    print(f"🔧 API_ID: {API_ID}")
    print(f"🔧 API_HASH: {'OK' if API_HASH else 'MISSING'}")
    print(
        f"🔧 SESSION_STRING: "
        f"{'OK' if SESSION_STRING else 'MISSING'}"
    )
    print(
        f"🔧 GEMINI: "
        f"{'OK' if ai_client else 'DISABLED'}"
    )

    # Запускаем HTTP-сервер
    await start_web_server()

    # Дополнительный self-ping
    asyncio.create_task(
        self_ping_loop()
    )

    # ========================================================
    # TELEGRAM
    # ========================================================

    try:

        print("🔌 Подключение к Telegram...")

        await client.start()

        # Проверяем, кто авторизован
        me = await client.get_me()

        print("=" * 60)
        print("✅ TELEGRAM УСПЕШНО ПОДКЛЮЧЕН")
        print(f"👤 Имя: {me.first_name}")
        print(f"🆔 ID: {me.id}")
        print(
            f"📛 Username: "
            f"@{me.username}"
            if me.username
            else "📛 Username: отсутствует"
        )
        print("=" * 60)

        print("👂 Ожидаю команды...")
        print("👉 Попробуй отправить: .ping")

        # Главный цикл Telethon
        await client.run_until_disconnected()

    except Exception as e:

        print("=" * 60)
        print("❌ КРИТИЧЕСКАЯ ОШИБКА TELEGRAM")
        print(
            f"Тип: {type(e).__name__}"
        )
        print(
            f"Ошибка: {e}"
        )
        print("=" * 60)

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("🛑 Остановлено пользователем.")

    except Exception as e:
        print(
            f"💥 Application crashed: "
            f"{type(e).__name__}: {e}"
        )
