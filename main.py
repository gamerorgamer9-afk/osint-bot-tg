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
# CONFIGURATION
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
    auto_reconnect=True,
    connection_retries=10,
    retry_delay=5,
)


# ============================================================
# GEMINI CLIENT
# ============================================================

ai_client = None

if GEMINI_API_KEY:
    ai_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# BOT STATE
# ============================================================

bot_state = {
    "afk": False,
    "afk_reason": "",
    "afk_messages": [],

    # None / catgirl / tsundere
    "mode": None,

    "spam_active": False,
    "spam_task": None,
}


# ============================================================
# MESSAGE CACHE
# ============================================================

msg_cache = deque(maxlen=500)


# ============================================================
# PERSONALITY SYSTEM
# ============================================================

def get_personality_instruction():
    mode = bot_state.get("mode")

    if mode == "catgirl":
        return """
Ты — милая аниме-котодевочка.

Твоя манера общения:
- дружелюбная;
- милая;
- немного игривая;
- естественная, без чрезмерного переигрывания.

Иногда можешь использовать "ня", "мяу", "мур" и похожие выражения,
но НЕ используй их в каждом предложении.

Не описывай свои действия в *звёздочках*, если пользователь этого не просит.

Главное — отвечай по существу и сохраняй смысл вопроса пользователя.
"""

    if mode == "tsundere":
        return """
Ты — персонаж цундере.

Твоя манера общения:
- немного дерзкая;
- слегка раздражённая;
- иногда застенчивая;
- иногда делаешь вид, что тебе всё равно.

Можешь иногда использовать типичные цундере-фразы вроде:
"б-бука", "не подумай, что я стараюсь ради тебя",
"я просто так ответила".

Но НЕ используй такие фразы постоянно.

Несмотря на характер, всегда отвечай полезно и по существу.
Не превращай каждый ответ в карикатуру.
"""

    return """
Отвечай обычным нейтральным стилем.

Будь:
- понятным;
- кратким;
- естественным;
- полезным.
"""


# ============================================================
# GEMINI NORMAL GENERATION
# ============================================================

async def fast_gemini_generate(
    prompt: str,
    system_instruction: str | None = None
) -> str:

    if not ai_client:
        return "❌ GEMINI_API_KEY не настроен."

    personality = get_personality_instruction()

    final_instruction = (
        system_instruction
        or "Отвечай максимально понятно и по существу."
    )

    final_instruction += "\n\n" + personality

    try:

        config = types.GenerateContentConfig(
            system_instruction=final_instruction,
            max_output_tokens=350,
            temperature=0.7,
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

        return f"❌ Ошибка Gemini: {str(e)[:500]}"


# ============================================================
# RENDER WEB SERVER
# ============================================================

async def handle_root(request):
    return web.Response(
        text="Telegram Userbot is running.",
        content_type="text/plain"
    )


async def handle_health(request):

    return web.json_response({
        "status": "ok",
        "telegram_connected": client.is_connected(),
        "gemini_configured": ai_client is not None,
        "mode": bot_state.get("mode"),
    })


async def start_web_server():

    app = web.Application()

    app.router.add_get("/", handle_root)
    app.router.add_get("/health", handle_health)

    runner = web.AppRunner(app)

    await runner.setup()

    site = web.TCPSite(
        runner,
        "0.0.0.0",
        PORT
    )

    await site.start()

    print(
        f"🌐 Web server запущен на порту {PORT}"
    )

    return runner


# ============================================================
# RENDER SELF PING
# ============================================================

async def self_ping_loop():

    if not RENDER_EXTERNAL_URL:

        print(
            "ℹ️ RENDER_EXTERNAL_URL не установлен. "
            "Self-ping отключён."
        )

        return

    print(
        f"🔄 Self-ping включён: "
        f"{RENDER_EXTERNAL_URL}"
    )

    async with ClientSession() as session:

        while True:

            try:

                await asyncio.sleep(600)

                async with session.get(
                    RENDER_EXTERNAL_URL,
                    timeout=20
                ) as response:

                    print(
                        f"🔄 Self-ping: HTTP "
                        f"{response.status}"
                    )

            except asyncio.CancelledError:
                break

            except Exception as e:

                print(
                    f"⚠️ Self-ping ошибка: "
                    f"{type(e).__name__}: {e}"
                )


# ============================================================
# MESSAGE LISTENER / DEBUG
# ============================================================

@client.on(events.NewMessage)
async def message_listener(event):

    try:

        text = event.raw_text or ""

        print(
            f"📩 MESSAGE | "
            f"chat={event.chat_id} | "
            f"outgoing={event.out} | "
            f"text={repr(text[:100])}"
        )

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

                    text = (
                        cached.text
                        or "[без текста]"
                    )

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
# .PING
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

        await event.edit(
            "🏓 Pinging..."
        )

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
# .HELP
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

        "🏓 `.ping` — проверить работу\n"
        "📚 `.help` — список команд\n\n"

        "💤 `.afk [причина]` — включить AFK\n"
        "🌅 `.unafk` — выключить AFK\n\n"

        "🧠 `.ai [текст]` — Gemini AI\n"
        "📝 `.remember [число]` — саммари чата\n"
        "🎭 `.clone` — скопировать стиль\n\n"

        "🐱 `.catgirl` — режим котодевочки\n"
        "😤 `.tsundere` — режим цундере\n"
        "🔄 `.reset` — обычный режим\n"
        "🎭 `.mode` — показать режим\n\n"

        "📨 `.spam [количество] [текст]`\n"
        "⏹ `.stop` — остановить spam"
    )

    try:

        await event.edit(
            help_text
        )

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

    print(
        f"💤 AFK ON: {reason}"
    )

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

@client.on(
    events.NewMessage(
        incoming=True
    )
)
async def afk_listener(event):

    if not bot_state["afk"]:
        return

    if not (
        event.is_private
        or event.mentioned
    ):
        return

    try:

        sender = await event.get_sender()

        name = (
            getattr(sender, "first_name", None)
            or getattr(sender, "username", None)
            or "Кто-то"
        )

        text = (
            event.raw_text
            or "[без текста]"
        )

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
# .AI
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.ai\s+(.+)"
    )
)
async def ai_handler(event):

    prompt = event.pattern_match.group(1)

    mode = bot_state.get("mode")

    print(
        f"🧠 AI request | "
        f"mode={mode} | "
        f"prompt={prompt[:100]}"
    )

    if not ai_client:

        await event.edit(
            "❌ `GEMINI_API_KEY` не настроен."
        )

        return

    personality = get_personality_instruction()

    try:

        await event.edit(
            "🧠 *Генерация...*"
        )

        response = (
            await ai_client.aio.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=personality,
                    max_output_tokens=400,
                    temperature=0.7,
                )
            )
        )

        full_text = ""

        last_update = time.time()

        async for chunk in response:

            if not chunk.text:
                continue

            full_text += chunk.text

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

            full_text = (
                "❌ Gemini не вернул текст."
            )

        await event.edit(
            full_text
        )

    except Exception as e:

        print(
            f"❌ AI error: "
            f"{type(e).__name__}: {e}"
        )

        await event.edit(
            f"❌ Ошибка Gemini:\n"
            f"`{str(e)[:500]}`"
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

    if command == "catgirl":

        bot_state["mode"] = "catgirl"

        await event.edit(
            "🐱 **Режим котодевочки активирован!**\n\n"
            "Теперь `.ai` будет отвечать "
            "в стиле милой аниме-котодевочки."
        )

        print(
            "🐱 MODE = CATGIRL"
        )

        return

    if command == "tsundere":

        bot_state["mode"] = "tsundere"

        await event.edit(
            "😤 **Режим цундере активирован!**\n\n"
            "Теперь `.ai` будет отвечать "
            "в стиле цундере."
        )

        print(
            "😤 MODE = TSUNDERE"
        )

        return

    if command == "reset":

        bot_state["mode"] = None

        await event.edit(
            "🔄 **Режим личности сброшен.**\n\n"
            "Теперь `.ai` использует обычный стиль."
        )

        print(
            "🔄 MODE = NORMAL"
        )


# ============================================================
# .MODE
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.mode$"
    )
)
async def mode_status(event):

    mode = bot_state.get("mode")

    if mode == "catgirl":

        text = (
            "🐱 Сейчас активен режим "
            "**котодевочки**."
        )

    elif mode == "tsundere":

        text = (
            "😤 Сейчас активен режим "
            "**цундере**."
        )

    else:

        text = (
            "🔄 Сейчас активен "
            "**обычный режим**."
        )

    await event.edit(
        text
    )


# ============================================================
# .REMEMBER
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

        limit = max(
            1,
            min(limit, 100)
        )

        await event.edit(
            f"📊 Анализирую последние "
            f"`{limit}` сообщений..."
        )

        raw_messages = await client.get_messages(
            event.chat_id,
            limit=limit
        )

        history = []

        for message in reversed(
            raw_messages
        ):

            if not message.text:
                continue

            try:

                sender = await message.get_sender()

                name = (
                    getattr(
                        sender,
                        "first_name",
                        None
                    )
                    or getattr(
                        sender,
                        "username",
                        None
                    )
                    or "User"
                )

            except Exception:

                name = "User"

            history.append(
                f"{name}: {message.text}"
            )

        if not history:

            await event.edit(
                "❌ Не удалось найти "
                "текстовые сообщения."
            )

            return

        context_text = "\n".join(
            history
        )

        prompt = (
            "Сделай краткое и понятное "
            "саммари этой переписки "
            "на русском языке.\n\n"
            f"{context_text}"
        )

        result = await fast_gemini_generate(
            prompt
        )

        await event.edit(
            f"📝 **Краткая выжимка:**\n\n"
            f"{result}"
        )

    except Exception as e:

        print(
            f"❌ Remember error: "
            f"{type(e).__name__}: {e}"
        )

        await event.edit(
            f"❌ Ошибка:\n"
            f"`{str(e)[:500]}`"
        )


# ============================================================
# .CLONE
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
                "❌ Используй `.clone` "
                "ответом на сообщение пользователя."
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
            message.text
            for message in messages
            if message.text
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
            "по следующим сообщениям.\n\n"
            "Затем напиши одно короткое "
            "предложение в похожей манере.\n\n"
            "Сохрани характерную пунктуацию, "
            "сленг и стиль.\n\n"
            f"{sample}"
        )

        result = await fast_gemini_generate(
            prompt
        )

        await event.edit(
            result
        )

    except Exception as e:

        print(
            f"❌ Clone error: "
            f"{type(e).__name__}: {e}"
        )

        await event.edit(
            f"❌ Ошибка:\n"
            f"`{str(e)[:500]}`"
        )


# ============================================================
# .SPAM
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

        # Ограничение
        count = max(
            1,
            min(count, 100)
        )

        # Останавливаем предыдущую задачу
        old_task = bot_state.get(
            "spam_task"
        )

        if old_task and not old_task.done():

            bot_state["spam_active"] = False

            old_task.cancel()

        await event.delete()

        bot_state["spam_active"] = True

        async def spam_loop():

            try:

                for _ in range(count):

                    if not bot_state[
                        "spam_active"
                    ]:
                        break

                    await client.send_message(
                        event.chat_id,
                        text
                    )

                    await asyncio.sleep(
                        0.5
                    )

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
# .STOP
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.stop$"
    )
)
async def stop_spam(event):

    bot_state["spam_active"] = False

    task = bot_state.get(
        "spam_task"
    )

    if task and not task.done():

        task.cancel()

    bot_state["spam_task"] = None

    await event.edit(
        "⏹ **Процесс остановлен.**"
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 60)
    print("🚀 ЗАПУСК TELEGRAM USERBOT")
    print("=" * 60)

    print(
        f"🔧 API_ID: {API_ID}"
    )

    print(
        "🔧 API_HASH: "
        f"{'OK' if API_HASH else 'MISSING'}"
    )

    print(
        "🔧 SESSION_STRING: "
        f"{'OK' if SESSION_STRING else 'MISSING'}"
    )

    print(
        "🔧 GEMINI: "
        f"{'OK' if ai_client else 'DISABLED'}"
    )

    # Render HTTP server
    await start_web_server()

    # Self-ping
    asyncio.create_task(
        self_ping_loop()
    )

    # Telegram
    try:

        print(
            "🔌 Подключение к Telegram..."
        )

        await client.start()

        me = await client.get_me()

        print("=" * 60)
        print(
            "✅ TELEGRAM УСПЕШНО ПОДКЛЮЧЕН"
        )

        print(
            f"👤 Имя: {me.first_name}"
        )

        print(
            f"🆔 ID: {me.id}"
        )

        if me.username:

            print(
                f"📛 Username: @{me.username}"
            )

        else:

            print(
                "📛 Username: отсутствует"
            )

        print("=" * 60)

        print(
            "👂 Ожидаю команды..."
        )

        print(
            "👉 Попробуй отправить: .ping"
        )

        await client.run_until_disconnected()

    except Exception as e:

        print("=" * 60)
        print(
            "❌ КРИТИЧЕСКАЯ ОШИБКА TELEGRAM"
        )

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

        print(
            "🛑 Остановлено пользователем."
        )

    except Exception as e:

        print(
            f"💥 Application crashed: "
            f"{type(e).__name__}: {e}"
        )
