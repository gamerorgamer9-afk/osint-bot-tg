import asyncio
import os
import re
import time
from collections import deque
from typing import Optional

from aiohttp import web
from telethon import TelegramClient, events
from telethon.sessions import StringSession

from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

API_ID = int(os.environ.get("API_ID", "0"))
API_HASH = os.environ.get("API_HASH", "")
SESSION_STRING = os.environ.get("SESSION_STRING", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

PORT = int(os.environ.get("PORT", "8080"))

# Можно поменять модели через Render Environment Variables.
#
# ВАЖНО: gemini-2.5-flash больше НЕ выдаётся новым ключам —
# Google возвращает 404 "no longer available to new users"
# и требует использовать gemini-3.6-flash.
#
# У Google квоты бесплатного тарифа считаются ОТДЕЛЬНО
# для каждой модели. Поэтому вместо одной модели используем
# ЦЕПОЧКУ: если у первой модели кончился лимит (429/quota),
# автоматически пробуем следующую — у неё свой независимый
# лимит, ещё не тронутый.
#
# Порядок: основная модель первая, дальше — более лёгкие
# "lite"-модели как запасной вариант.
#
# Переопределить можно через GEMINI_MODELS
# (модели через запятую).
GEMINI_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_MODELS",
        "gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
    ).split(",")
    if m.strip()
]

# Оставляем и старую переменную для обратной совместимости —
# если задана вручную, ставим её первой в цепочке.
_legacy_model = os.environ.get("GEMINI_MODEL")

if _legacy_model and _legacy_model not in GEMINI_MODELS:
    GEMINI_MODELS.insert(0, _legacy_model)


# ============================================================
# VALIDATION
# ============================================================

if not API_ID:
    raise RuntimeError("API_ID не установлен!")

if not API_HASH:
    raise RuntimeError("API_HASH не установлен!")

if not SESSION_STRING:
    raise RuntimeError("SESSION_STRING не установлен!")

if not GEMINI_API_KEY:
    print("⚠️ GEMINI_API_KEY не установлен. AI-функции будут отключены.")


# ============================================================
# TELEGRAM CLIENT
# ============================================================

# ВАЖНО:
# Используем именно StringSession.
# Нельзя делать:
#
# client.start(phone=lambda: SESSION_STRING)
#
# SESSION_STRING — это уже авторизованная String Session.

client = TelegramClient(
    StringSession(SESSION_STRING),
    API_ID,
    API_HASH,
    connection_retries=5,
    retry_delay=2
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

    # Текущая личность:
    # None
    # catgirl
    # tsundere
    "mode": None,

    # AFK
    "afk": False,
    "afk_reason": "",
    "afk_messages": [],

    # Spam
    "spam_active": False,

    # Mute:
    # chat_id -> True
    #
    # Работает только для ЛС.
    "muted_chats": set(),

}


# ============================================================
# MESSAGE CACHE
# ============================================================

# Храним последние сообщения для логирования удалений.
#
# 1500 сообщений достаточно для быстрого отслеживания,
# но память при этом почти не расходуется.

message_cache = deque(
    maxlen=1500
)


# ============================================================
# GEMINI SEMAPHORE
# ============================================================

# Не позволяем бесконечному количеству запросов
# одновременно улетать в Gemini.

gemini_semaphore = asyncio.Semaphore(1)


# ============================================================
# HTTP SERVER FOR RENDER
# ============================================================

async def handle_ping(request):
    return web.Response(
        text="OK - Telegram userbot is running"
    )


async def start_web_server():

    app = web.Application()

    app.router.add_get(
        "/",
        handle_ping
    )

    app.router.add_get(
        "/health",
        handle_ping
    )

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


# ============================================================
# PERSONALITY PROMPTS
# ============================================================

def get_rewrite_personality():

    mode = bot_state["mode"]


    # ========================================================
    # LINIA DEDOLDIA
    # ========================================================

    if mode == "catgirl":

        return """
Ты переписываешь сообщение пользователя
в манере Линии Дедолдии (Linia Dedoldia)
из "Mushoku Tensei: Jobless Reincarnation".

Ты НЕ отвечаешь пользователю.

Ты только переписываешь его исходное сообщение.


ХАРАКТЕР:

Линия:

- самоуверенная;
- дерзкая;
- хвастливая;
- импульсивная;
- эмоциональная;
- прямолинейная;
- немного вспыльчивая;
- любит показывать своё превосходство;
- может быть язвительной;
- иногда ведёт себя по-детски;
- уважает действительно сильных людей.

Не делай её постоянно милой.

Не делай её постоянно злой.

Основное впечатление:
самоуверенная, наглая, эмоциональная
кошачья девушка.


МАНЕРА РЕЧИ:

Речь должна быть:

- разговорной;
- живой;
- эмоциональной;
- дерзкой;
- немного кошачьей;
- естественной.

Используй "ня" умеренно.

Не добавляй "ня" в каждое предложение.


КОШАЧЬИ ИСКАЖЕНИЯ:

Иногда можешь слегка искажать
отдельные обычные разговорные слова,
добавляя кошачьи элементы.

Например, допустимы естественные
небольшие искажения вроде добавления
"мяу" или окончания "ня".

Используй такие искажения редко.

В среднем:
0–1 необычное кошачье искажение
на сообщение.

Иногда вообще не используй искажения.

Не превращай каждое слово
в кошачью версию.


НЕ ДЕЛАЙ:

Не превращай речь в карикатурную
аниме-котодевочку.

Не используй постоянно:

"мяу"

"мяу-мяу"

"мур"

"мур-мур"

"няшечка"

"няшка"

"uwu"

"nya nya"

Кошачьи особенности должны быть
небольшой деталью характера.


СМЫСЛ:

Полностью сохраняй смысл
исходного сообщения.

Не добавляй новые факты.

Не придумывай события.

Не меняй намерение пользователя.

Не делай сообщение намного длиннее.


ТЕХНИЧЕСКИЕ ДАННЫЕ:

Никогда не изменяй:

- имена;
- названия;
- usernames;
- ссылки;
- числа;
- даты;
- команды;
- ID;
- технические термины;
- код.


ФОРМАТ ОТВЕТА:

Верни ТОЛЬКО переписанное сообщение.

Без объяснений.

Без кавычек.

Без слов вроде:
"Вот версия..."

Без упоминания Gemini.

Без упоминания ИИ.

Без описания своих действий.
"""


    # ========================================================
    # TSUNDERE
    # ========================================================

    if mode == "tsundere":

        return """
Ты переписываешь сообщение пользователя
в естественной манере цундере.

Ты НЕ отвечаешь пользователю.

Ты только переписываешь
его исходное сообщение.


ХАРАКТЕР:

- самоуверенная;
- немного дерзкая;
- эмоциональная;
- вспыльчивая;
- иногда смущённая;
- делает вид, что ей всё равно;
- может скрывать заботу;
- иногда подкалывает собеседника;
- может отрицать собственную симпатию.

Не делай её постоянно злой.

Не делай её постоянно милой.


МАНЕРА РЕЧИ:

Используй естественную разговорную речь.

Иногда допустимы короткие эмоциональные
вставки вроде:

"Пф!"

"Бака!"

"Не то чтобы я..."

"Я просто..."

"Не думай ничего такого!"

Но используй их редко.

Не превращай каждое сообщение
в аниме-клише.


СМЫСЛ:

Полностью сохраняй смысл
исходного сообщения.

Не добавляй новые факты.

Не придумывай события.

Не меняй намерение пользователя.

Не делай сообщение намного длиннее.


ТЕХНИЧЕСКИЕ ДАННЫЕ:

Никогда не изменяй:

- имена;
- названия;
- usernames;
- ссылки;
- числа;
- даты;
- команды;
- ID;
- технические термины;
- код.


ФОРМАТ ОТВЕТА:

Верни ТОЛЬКО переписанное сообщение.

Без объяснений.

Без кавычек.

Без слов:
"Вот версия..."

Не упоминай Gemini.

Не упоминай ИИ.
"""


    return None


# ============================================================
# CLEAN GEMINI OUTPUT
# ============================================================

def clean_ai_text(text: str) -> str:

    if not text:
        return ""

    text = text.strip()

    # Убираем случайные кавычки,
    # если Gemini решил обернуть ответ.
    if (
        len(text) >= 2
        and text.startswith('"')
        and text.endswith('"')
    ):
        text = text[1:-1].strip()

    return text


# ============================================================
# RETRY ON 429 (RATE LIMIT)
# ============================================================

def _is_rate_limit_error(e: Exception) -> bool:

    msg = str(e)

    return (
        "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "quota" in msg.lower()
    )


def _is_model_unavailable_error(e: Exception) -> bool:

    # Модель может быть недоступна конкретно этому ключу
    # ("no longer available to new users") или не существовать
    # под этим именем — в обоих случаях есть смысл просто
    # попробовать следующую модель в цепочке, а не сразу
    # сдаваться.

    msg = str(e)

    return (
        "404" in msg
        or "NOT_FOUND" in msg
        or "not found" in msg.lower()
        or "no longer available" in msg.lower()
    )


def _should_try_next_model(e: Exception) -> bool:

    return (
        _is_rate_limit_error(e)
        or _is_model_unavailable_error(e)
    )


def _extract_retry_delay(e: Exception) -> Optional[float]:

    # Google обычно пишет прямо в тексте ошибки, сколько
    # секунд ждать, например: "Please retry in 16.67s".
    # Если нашли — используем это число, оно точнее,
    # чем наш собственный экспоненциальный backoff.

    match = re.search(
        r"retry in ([\d.]+)s",
        str(e)
    )

    if match:

        try:
            return float(match.group(1)) + 1.0
        except ValueError:
            return None

    return None


def _format_gemini_error(e: Exception) -> str:

    # Собираем понятное сообщение об ошибке.
    #
    # Если это лимит 429 — явно показываем, сколько ждать,
    # т.к. это число часто "отрезается" при обрезке текста
    # ошибки до 500 символов.

    if _is_rate_limit_error(e):

        delay = _extract_retry_delay(e)

        if delay:

            return (
                "⏳ Превышен лимит запросов к Gemini "
                f"(429). Попробуй снова через "
                f"~{delay:.0f} сек."
            )

        return (
            "⏳ Превышен лимит запросов к Gemini (429). "
            "Не удалось определить точное время ожидания "
            "из ответа — обычно это 20–60 секунд "
            "(при исчерпании дневной квоты — "
            "до полуночи по Тихоокеанскому времени США)."
        )

    return f"`{str(e)[:500]}`"


async def _with_retry(coro_factory, retries: int = 3, base_delay: float = 5.0):

    # coro_factory — функция БЕЗ аргументов, возвращающая
    # новую корутину при каждом вызове (нужно для повторных попыток).

    last_error = None

    for attempt in range(retries + 1):

        try:

            return await coro_factory()

        except Exception as e:

            last_error = e

            if not _is_rate_limit_error(e) or attempt == retries:
                raise

            delay = (
                _extract_retry_delay(e)
                or base_delay * (2 ** attempt)
            )

            print(
                f"⏳ Gemini 429, повтор через {delay:.0f}с "
                f"(попытка {attempt + 1}/{retries})"
            )

            await asyncio.sleep(delay)

    raise last_error


# ============================================================
# FAST GEMINI GENERATION
# ============================================================

async def gemini_generate(
    prompt: str,
    system_instruction: str,
    max_tokens: int = 180
):

    if not ai_client:
        raise RuntimeError(
            "GEMINI_API_KEY не установлен"
        )

    config = types.GenerateContentConfig(

        system_instruction=system_instruction,

        max_output_tokens=max_tokens,

        temperature=0.35,

        top_p=0.9,

    )

    last_error = None

    for model in GEMINI_MODELS:

        async def _call(model=model):

            async with gemini_semaphore:

                return await asyncio.wait_for(

                    ai_client.aio.models.generate_content(

                        model=model,

                        contents=prompt,

                        config=config

                    ),

                    timeout=12

                )

        try:

            # На каждую модель — по 1 повтору, чтобы не
            # тратить всё время на одну модель, если рядом
            # есть ещё две с чистой квотой.
            response = await _with_retry(
                _call,
                retries=1,
                base_delay=3.0
            )

            return clean_ai_text(
                response.text or ""
            )

        except Exception as e:

            last_error = e

            if not _should_try_next_model(e):
                raise

            reason = (
                "исчерпал лимит"
                if _is_rate_limit_error(e)
                else "недоступна для этого ключа"
            )

            print(
                f"↪️ {model} {reason}, "
                "пробуем следующую модель..."
            )

    raise last_error


# ============================================================
# FAST STREAMING GEMINI
# ============================================================

async def gemini_stream(
    prompt: str,
    system_instruction: str,
    max_tokens: int = 180
):

    if not ai_client:
        raise RuntimeError(
            "GEMINI_API_KEY не установлен"
        )

    config = types.GenerateContentConfig(

        system_instruction=system_instruction,

        max_output_tokens=max_tokens,

        temperature=0.35,

        top_p=0.9,

    )

    stream = None
    last_error = None

    for model in GEMINI_MODELS:

        async def _open_stream(model=model):

            async with gemini_semaphore:

                return await asyncio.wait_for(

                    ai_client.aio.models.generate_content_stream(

                        model=model,

                        contents=prompt,

                        config=config

                    ),

                    timeout=12

                )

        try:

            # Ретраим только открытие стрима (сам 429
            # прилетает именно на этом шаге, до получения
            # первых чанков).
            stream = await _with_retry(
                _open_stream,
                retries=1,
                base_delay=3.0
            )

            break

        except Exception as e:

            last_error = e

            if not _should_try_next_model(e):
                raise

            reason = (
                "исчерпал лимит"
                if _is_rate_limit_error(e)
                else "недоступна для этого ключа"
            )

            print(
                f"↪️ {model} {reason}, "
                "пробуем следующую модель..."
            )

    if stream is None:
        raise last_error

    full_text = ""

    async for chunk in stream:

        try:
            piece = chunk.text
        except Exception:
            piece = None

        if not piece:
            continue

        full_text += piece

        yield clean_ai_text(full_text)


# ============================================================
# MESSAGE CACHE
# ============================================================

@client.on(events.NewMessage)
async def cache_listener(event):

    try:

        message = event.message

        if not message:
            return

        # Кэшируем текстовые сообщения.
        if message.text:

            message_cache.append({

                "id": message.id,

                "chat_id": event.chat_id,

                "sender_id": message.sender_id,

                "text": message.text,

                "date": message.date,

            })

    except Exception as e:

        print(
            f"⚠️ Cache error: {e}"
        )


# ============================================================
# DELETED MESSAGE LOGGER
# ============================================================

@client.on(events.MessageDeleted)
async def deleted_logger(event):

    try:

        for deleted_id in event.deleted_ids:

            found = None

            for cached in reversed(message_cache):

                if cached["id"] == deleted_id:

                    found = cached

                    break

            if not found:
                continue


            # ------------------------------------------------
            # Получаем название чата
            # ------------------------------------------------

            chat_name = "Неизвестный чат"

            try:

                chat = await client.get_entity(
                    found["chat_id"]
                )

                if getattr(chat, "title", None):

                    chat_name = chat.title

                elif getattr(chat, "first_name", None):

                    chat_name = chat.first_name

                    if getattr(
                        chat,
                        "last_name",
                        None
                    ):
                        chat_name += (
                            f" {chat.last_name}"
                        )

                elif getattr(chat, "username", None):

                    chat_name = (
                        f"@{chat.username}"
                    )

            except Exception:

                chat_name = (
                    f"Chat ID: {found['chat_id']}"
                )


            # ------------------------------------------------
            # Получаем имя отправителя
            # ------------------------------------------------

            sender_name = "Неизвестно"

            try:

                sender = await client.get_entity(
                    found["sender_id"]
                )

                if getattr(sender, "first_name", None):

                    sender_name = sender.first_name

                    if getattr(
                        sender,
                        "last_name",
                        None
                    ):
                        sender_name += (
                            f" {sender.last_name}"
                        )

                elif getattr(sender, "username", None):

                    sender_name = (
                        f"@{sender.username}"
                    )

            except Exception:
                pass


            # ------------------------------------------------
            # Лог
            # ------------------------------------------------

            log_text = (
                "🗑 **Удалено сообщение**\n\n"

                f"👤 **От:** {sender_name}\n"

                f"🆔 **Sender ID:** "
                f"`{found['sender_id']}`\n\n"

                f"📍 **Где:** {chat_name}\n"

                f"🆔 **Chat ID:** "
                f"`{found['chat_id']}`\n\n"

                f"💬 **Текст:**\n"
                f"{found['text']}"
            )


            try:

                await client.send_message(
                    "me",
                    log_text
                )

            except Exception as e:

                print(
                    f"⚠️ Не удалось сохранить "
                    f"удалённое сообщение: {e}"
                )


    except Exception as e:

        print(
            f"❌ Deleted logger error: {e}"
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

    start = time.perf_counter()

    await event.edit(
        "🏓 Проверяю..."
    )

    ms = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    await event.edit(
        f"🚀 **Pong!**\n"
        f"⚡ Telegram: `{ms} ms`"
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

    text = """
⚙️ **Команды юзербота**

🏓 `.ping`
Проверка Telegram.

🧠 `.ai текст`
Gemini без выбранной личности.

🎭 `.catgirl`
Стиль Линии Дедолдии.

🔥 `.tsundere`
Стиль цундере.

🔄 `.reset`
Отключить личность.

🎭 `.clone`
Клонирование стиля автора сообщения.
Не использует текущую личность.

💤 `.afk [причина]`
Включить AFK.

🌅 `.unafk`
Выключить AFK.

🔇 `.mute`
Удалять следующие сообщения собеседника в ЛС.

🔊 `.unmute`
Выключить mute.

📨 `.spam количество текст`
Запустить spam.

⛔ `.off`
Принудительно остановить spam.

🗑 Удалённые сообщения
Сохраняются в Избранные с указанием чата.
"""

    await event.edit(text)


# ============================================================
# PERSONALITY COMMAND
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.(catgirl|tsundere|reset)$"
    )
)
async def personality_handler(event):

    command = event.pattern_match.group(1)


    if command == "reset":

        bot_state["mode"] = None

        await event.edit(
            "🔄 **Режим личности отключён.**"
        )

        print(
            "🎭 Personality: OFF"
        )

        return


    bot_state["mode"] = command


    if command == "catgirl":

        await event.edit(
            "🐱 **Режим Линии Дедолдии включён.**"
        )

        print(
            "🐱 Personality: LINIA"
        )

    elif command == "tsundere":

        await event.edit(
            "🔥 **Режим цундере включён.**"
        )

        print(
            "🔥 Personality: TSUNDERE"
        )


# ============================================================
# AFK
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

    await event.edit(
        f"💤 **AFK включён**\n"
        f"Причина: `{reason}`"
    )


@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.unafk$"
    )
)
async def afk_off(event):

    if not bot_state["afk"]:

        await event.edit(
            "❌ AFK сейчас выключен."
        )

        return


    count = len(
        bot_state["afk_messages"]
    )


    bot_state["afk"] = False

    bot_state["afk_reason"] = ""

    bot_state["afk_messages"] = []


    await event.edit(
        f"🌅 **AFK выключен.**\n"
        f"Сообщений во время AFK: `{count}`"
    )


# ============================================================
# AFK LISTENER
# ============================================================

@client.on(events.NewMessage(incoming=True))
async def afk_listener(event):

    if not bot_state["afk"]:
        return

    if not (
        event.is_private
        or event.mentioned
    ):
        return


    sender = await event.get_sender()

    name = (
        getattr(
            sender,
            "first_name",
            None
        )
        or "Кто-то"
    )


    bot_state["afk_messages"].append({

        "name": name,

        "text": event.text or ""

    })


    try:

        await event.reply(
            "🤖 Я сейчас AFK.\n"
            f"Причина: `{bot_state['afk_reason']}`"
        )

    except Exception as e:

        print(
            f"⚠️ AFK reply error: {e}"
        )


# ============================================================
# MUTE
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.mute$"
    )
)
async def mute_handler(event):

    if not event.is_private:

        await event.edit(
            "❌ `.mute` работает только в ЛС."
        )

        return


    bot_state["muted_chats"].add(
        event.chat_id
    )


    await event.edit(
        "🔇 **Mute включён.**\n"
        "Следующие сообщения собеседника "
        "в этом ЛС будут удаляться."
    )


# ============================================================
# UNMUTE
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.unmute$"
    )
)
async def unmute_handler(event):

    if not event.is_private:

        await event.edit(
            "❌ `.unmute` работает только в ЛС."
        )

        return


    bot_state["muted_chats"].discard(
        event.chat_id
    )


    await event.edit(
        "🔊 **Mute выключен.**"
    )


# ============================================================
# MUTE LISTENER
# ============================================================

@client.on(events.NewMessage(incoming=True))
async def mute_listener(event):

    if not event.is_private:
        return


    if event.chat_id not in bot_state[
        "muted_chats"
    ]:
        return


    try:

        await event.delete()

        print(
            f"🔇 Удалено сообщение "
            f"из muted ЛС: {event.chat_id}"
        )

    except Exception as e:

        print(
            f"⚠️ Mute delete error: {e}"
        )


# ============================================================
# AI COMMAND
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.ai\s+(.+)"
    )
)
async def ai_handler(event):

    prompt = event.pattern_match.group(1)


    if not ai_client:

        await event.edit(
            "❌ GEMINI_API_KEY не установлен."
        )

        return


    await event.edit(
        "🧠 Генерирую..."
    )


    system_prompt = """
Ты полезный AI-ассистент.

Отвечай кратко, понятно и по существу.

Не используй никакие выбранные
пользователем ролевые личности.

Не используй стиль Линии.

Не используй стиль цундере.

Не добавляй "ня", "мяу" и подобные слова,
если пользователь специально этого не попросил.

Отвечай только на запрос пользователя.
"""


    try:

        # ----------------------------------------------------
        # STREAMING
        # ----------------------------------------------------

        current_text = ""

        first_update = True

        last_edit = 0.0


        async for partial in gemini_stream(
            prompt,
            system_prompt,
            max_tokens=400
        ):

            if not partial:
                continue


            current_text = partial


            now = time.monotonic()


            # Первое обновление сразу.
            #
            # Последующие не чаще ~0.7 сек,
            # чтобы не получить FloodWait.
            if (
                first_update
                or now - last_edit >= 0.7
            ):

                await event.edit(
                    current_text + " ▌"
                )

                first_update = False

                last_edit = now


        if current_text:

            await event.edit(
                current_text
            )

        else:

            await event.edit(
                "❌ Gemini не вернул текст."
            )


    except Exception as e:

        print(
            f"❌ Gemini error: {repr(e)}"
        )

        await event.edit(
            "❌ Ошибка Gemini:\n"
            + _format_gemini_error(e)
        )


# ============================================================
# PERSONALITY REWRITER
# ============================================================

@client.on(events.NewMessage(outgoing=True))
async def personality_rewriter(event):

    # --------------------------------------------------------
    # Нет активной личности
    # --------------------------------------------------------

    if bot_state["mode"] is None:
        return


    # --------------------------------------------------------
    # Только текст
    # --------------------------------------------------------

    if not event.text:
        return


    text = event.text.strip()


    if not text:
        return


    # --------------------------------------------------------
    # НЕ обрабатываем команды
    # --------------------------------------------------------

    if text.startswith("."):
        return


    # --------------------------------------------------------
    # Получаем текущий промпт
    # --------------------------------------------------------

    system_prompt = get_rewrite_personality()


    if not system_prompt:
        return


    original_mode = bot_state["mode"]


    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    try:

        # Сразу показываем обработку.
        #
        # Это почти не влияет на скорость,
        # но пользователь видит, что сообщение принято.

        await event.edit(
            text + " ▌"
        )


        current_text = ""

        first_update = True

        last_edit = 0.0


        async for partial in gemini_stream(
            text,
            system_prompt,
            max_tokens=180
        ):

            if not partial:
                continue


            current_text = partial


            now = time.monotonic()


            # ------------------------------------------------
            # Очень быстрое первое обновление
            # ------------------------------------------------

            if first_update:

                await event.edit(
                    current_text + " ▌"
                )

                first_update = False

                last_edit = now

                continue


            # ------------------------------------------------
            # Последующие обновления
            # ------------------------------------------------

            if now - last_edit >= 0.7:

                await event.edit(
                    current_text + " ▌"
                )

                last_edit = now


        # ----------------------------------------------------
        # Если режим не изменился во время генерации,
        # публикуем результат.
        #
        # Если пользователь успел сделать .reset
        # или сменить личность — старый запрос всё равно
        # заканчивается, но сообщение не будет дополнительно
        # обрабатываться.
        # ----------------------------------------------------

        if current_text:

            await event.edit(
                current_text
            )

        else:

            await event.edit(
                text
            )


    except Exception as e:

        print(
            f"❌ Personality error "
            f"({original_mode}): {repr(e)}"
        )

        # При ошибке возвращаем оригинальное сообщение,
        # чтобы оно не оставалось с "▌".

        try:

            await event.edit(
                text
            )

        except Exception:
            pass

        # Раньше ошибка была видна только в логах Render,
        # из-за чего казалось, что .catgirl/.tsundere
        # "просто не работают". Теперь дублируем её
        # в Избранное, чтобы было видно прямо в Telegram.

        try:

            await client.send_message(
                "me",
                "⚠️ **Personality rewriter error**\n"
                f"Mode: `{original_mode}`\n"
                + _format_gemini_error(e)
            )

        except Exception:
            pass


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

    reply = await event.get_reply_message()


    if not reply or not reply.sender_id:

        await event.edit(
            "❌ Используй `.clone` "
            "ответом на сообщение пользователя."
        )

        return


    await event.edit(
        "🎭 Анализирую стиль..."
    )


    try:

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
                "❌ Недостаточно сообщений."
            )

            return


        sample = "\n".join(
            texts[:15]
        )


        prompt = f"""
Проанализируй стиль автора сообщений
и напиши ОДНО короткое сообщение
в похожей манере.

Не отвечай на содержимое сообщений.

Не используй личность пользователя.

Не используй режим catgirl.

Не используй tsundere.

Не добавляй "ня" или "мяу"
только потому, что у пользователя
может быть активирован другой режим.

Верни только одно сообщение.

Примеры сообщений автора:

{sample}
"""


        system_prompt = """
Ты анализируешь стиль текста.

Твоя задача — написать короткое сообщение,
похожее по стилю, лексике, длине,
пунктуации и эмоциональной манере
на предоставленные сообщения.

Не копируй сообщение дословно.

Не добавляй объяснения.
"""


        result = await gemini_generate(
            prompt,
            system_prompt,
            max_tokens=120
        )


        await event.edit(
            result or "❌ Gemini не вернул результат."
        )


    except Exception as e:

        print(
            f"❌ Clone error: {repr(e)}"
        )

        await event.edit(
            "❌ Ошибка: "
            + _format_gemini_error(e)
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

        limit = max(
            1,
            min(limit, 100)
        )


        await event.edit(
            f"📊 Анализирую "
            f"последние {limit} сообщений..."
        )


        messages = await client.get_messages(
            event.chat_id,
            limit=limit
        )


        history = []


        for message in reversed(messages):

            if not message.text:
                continue


            name = "User"


            try:

                sender = await message.get_sender()

                if getattr(
                    sender,
                    "first_name",
                    None
                ):

                    name = sender.first_name

            except Exception:
                pass


            history.append(
                f"{name}: {message.text}"
            )


        if not history:

            await event.edit(
                "❌ Нет текстовых сообщений."
            )

            return


        context = "\n".join(
            history
        )


        prompt = f"""
Сделай краткое и понятное саммари
этой переписки.

Не используй личность Линии.

Не используй цундере.

Не добавляй ролевой стиль.

Переписка:

{context}
"""


        result = await gemini_generate(
            prompt,
            "Отвечай кратко и структурированно.",
            max_tokens=300
        )


        await event.edit(
            "📝 **Краткая выжимка:**\n\n"
            + result
        )


    except Exception as e:

        print(
            f"❌ Remember error: {repr(e)}"
        )

        await event.edit(
            "❌ Ошибка: "
            + _format_gemini_error(e)
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

    count = int(
        event.pattern_match.group(1)
    )

    text = event.pattern_match.group(2)


    # Ограничение от случайного
    # огромного количества сообщений.
    count = max(
        1,
        min(count, 100)
    )


    await event.delete()


    # Останавливаем предыдущий spam.
    bot_state["spam_active"] = False

    await asyncio.sleep(0.05)


    bot_state["spam_active"] = True


    print(
        f"📨 Spam started: {count}"
    )


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


            # 0.3 сек между сообщениями.
            await asyncio.sleep(
                0.3
            )


    except Exception as e:

        print(
            f"❌ Spam error: {repr(e)}"
        )


    finally:

        bot_state["spam_active"] = False


        print(
            "📨 Spam finished"
        )


# ============================================================
# OFF
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.off$"
    )
)
async def off_handler(event):

    bot_state["spam_active"] = False


    await event.edit(
        "⛔ **Spam принудительно остановлен.**"
    )


# ============================================================
# STARTUP
# ============================================================

async def main():

    print(
        "=========================================="
    )

    print(
        "🚀 Telegram Userbot starting..."
    )

    print(
        f"🧠 Gemini модели (цепочка): {' → '.join(GEMINI_MODELS)}"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # Web server
    # --------------------------------------------------------

    await start_web_server()


    # --------------------------------------------------------
    # Telegram
    # --------------------------------------------------------

    print(
        "🔐 Подключение к Telegram..."
    )


    await client.connect()


    if not await client.is_user_authorized():

        raise RuntimeError(
            "SESSION_STRING недействителен "
            "или Telegram-сессия не авторизована."
        )


    me = await client.get_me()


    print(
        "=========================================="
    )

    print(
        "✅ USERBOT УСПЕШНО ЗАПУЩЕН"
    )

    print(
        f"👤 Имя: "
        f"{getattr(me, 'first_name', 'Unknown')}"
    )

    print(
        f"🆔 ID: {me.id}"
    )

    print(
        f"📱 Username: "
        f"@{me.username}"
        if me.username
        else "📱 Username: отсутствует"
    )

    print(
        "=========================================="
    )


    # --------------------------------------------------------
    # Проверяем Gemini
    # --------------------------------------------------------

    if ai_client:

        print(
            "🧠 Gemini: включён, "
            f"цепочка моделей: {' → '.join(GEMINI_MODELS)}"
        )

        try:

            test_result = await gemini_generate(
                "Скажи одно слово: тест",
                "Отвечай одним словом.",
                max_tokens=10
            )

            print(
                f"✅ Gemini self-test OK: {test_result!r}"
            )

        except Exception as e:

            print(
                "❌ Gemini self-test FAILED "
                f"(все модели из цепочки не сработали): "
                f"{repr(e)}"
            )

            print(
                "   Проверь: правильный ли GEMINI_API_KEY, "
                "доступна ли хотя бы одна из моделей "
                f"{GEMINI_MODELS} для этого ключа, "
                "и не блокирует ли Render исходящие "
                "запросы к generativelanguage.googleapis.com."
            )

    else:

        print(
            "⚠️ Gemini: выключен "
            "(GEMINI_API_KEY не установлен в Render Environment)"
        )


    print(
        "👂 Ожидаю сообщения..."
    )


    await client.run_until_disconnected()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "🛑 Userbot stopped."
        )

    except Exception as e:

        print(
            f"💥 FATAL ERROR: {repr(e)}"
        )
