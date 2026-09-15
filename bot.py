import asyncio
import ast
import io
import logging
import operator
import os
import random
import re
import time
from collections import deque, OrderedDict
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from openai import AsyncOpenAI
from telethon import TelegramClient, events, functions, types
from telethon.errors import FloodWaitError
from telethon.sessions import StringSession

try:
    from aiohttp import web as aiohttp_web
except ImportError:
    aiohttp_web = None

# Дополнительные (опциональные) зависимости под .ocr/.vm — требуют
# ещё и системных бинарников (Tesseract OCR, ffmpeg), поэтому
# оборачиваем в try/except: если не установлены, бот всё равно
# стартует, просто эти конкретные команды сообщат, чего не хватает.
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import pytesseract

    _TESSERACT_CMD = os.getenv("TESSERACT_CMD")
    if _TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
except ImportError:
    pytesseract = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None


# ============================================================
# НАСТРОЙКИ
# ============================================================

load_dotenv()

API_ID = int(os.getenv("TG_API_ID", "0"))
API_HASH = os.getenv("TG_API_HASH", "")
TG_SESSION = os.getenv("TG_SESSION", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# deepseek-chat / deepseek-reasoner выведены из обращения —
# актуальная модель: deepseek-v4-flash.
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)

# У моделей семейства DeepSeek-V4 thinking-режим включён по
# умолчанию для любого имени модели. Он тратит часть max_tokens
# на внутренние рассуждения и заставляет API игнорировать
# temperature. Для наших задач (переписать текст, вытащить
# факты) рассуждения не нужны — по умолчанию выключены явно.
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() in {
    "1", "true", "yes"
}

THINKING_EXTRA_BODY = {
    "thinking": {
        "type": "enabled" if ENABLE_THINKING else "disabled"
    }
}

CATGIRL_MIN_DELAY = float(os.getenv("CATGIRL_MIN_DELAY", "1"))
CATGIRL_MAX_DELAY = float(os.getenv("CATGIRL_MAX_DELAY", "5"))

REMEMBER_MAX_MESSAGES = int(os.getenv("REMEMBER_MAX_MESSAGES", "50000"))
PROCESSED_MESSAGES_MAX = int(os.getenv("PROCESSED_MESSAGES_MAX", "5000"))
FLOODWAIT_MAX_RETRIES = int(os.getenv("FLOODWAIT_MAX_RETRIES", "5"))

# Сколько раз повторять запрос к DeepSeek, если ответ пустой
# или запрос упал с ошибкой.
STYLE_REWRITE_MAX_ATTEMPTS = int(
    os.getenv("STYLE_REWRITE_MAX_ATTEMPTS", "3")
)

# На сколько токенов увеличивать max_tokens с каждой повторной
# попыткой при пустом ответе.
EMPTY_RESPONSE_TOKEN_STEP = int(
    os.getenv(
        "EMPTY_RESPONSE_TOKEN_STEP",
        "2000" if ENABLE_THINKING else "300",
    )
)

# Стартовый max_tokens для первой попытки.
DEFAULT_BASE_MAX_TOKENS = int(
    os.getenv(
        "DEFAULT_BASE_MAX_TOKENS",
        "3000" if ENABLE_THINKING else "600",
    )
)

CLONE_MAX_MESSAGES = int(os.getenv("CLONE_MAX_MESSAGES", "300"))
CLONE_MAX_CHARS = int(os.getenv("CLONE_MAX_CHARS", "6000"))

# Сколько символов сообщений собеседника отдаём DeepSeek для
# шуточной dere-классификации в конце .remember.
DERE_MAX_CHARS = int(os.getenv("DERE_MAX_CHARS", "6000"))

# --- Настройки токен-экономии .remember ------------------------
# Раньше .remember резал историю на куски по 12000 симв. и делал
# до 30 отдельных запросов на анализ + ещё 2 финальных (профиль,
# dere) — до 32 запросов на одну команду, и каждый заново тащит
# системную инструкцию. Теперь куски крупнее (меньше повторов
# инструкции на тот же объём текста), а профиль и dere-тип
# считаются ОДНИМ финальным запросом вместо двух. Суммарный
# потолок анализируемого текста тот же (кусков * размер куска),
# просто достигается за куда меньшее число обращений к API.
REMEMBER_CHUNK_CHARS = int(os.getenv("REMEMBER_CHUNK_CHARS", "45000"))
REMEMBER_MAX_CHUNKS = int(os.getenv("REMEMBER_MAX_CHUNKS", "8"))

TELEGRAM_MESSAGE_LIMIT = 4096

# --- .f (анимированный текст) ---------------------------------
# Задержка между шагами появления текста и максимум слов —
# защита от лишних edit'ов на очень длинный текст (FloodWait).
ANIMATED_TEXT_DELAY = float(os.getenv("ANIMATED_TEXT_DELAY", "0.4"))
ANIMATED_TEXT_MAX_WORDS = int(os.getenv("ANIMATED_TEXT_MAX_WORDS", "60"))

# --- .geo / .geolive (геокодинг через OpenStreetMap Nominatim) -
# Nominatim требует свой User-Agent и не более ~1 запроса/сек —
# для личного использования этого более чем достаточно.
NOMINATIM_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT", "TelegramUserbot/1.0 (personal use)"
)
GEOLIVE_PERIOD_SECONDS = int(os.getenv("GEOLIVE_PERIOD_SECONDS", "900"))

# --- .ocr (текст + QR/штрихкоды с фото) ------------------------
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "rus+eng")

# --- .vm (расшифровка голосовых/кружков) ------------------------
VOICE_TRANSCRIBE_LANG = os.getenv("VOICE_TRANSCRIBE_LANG", "ru-RU")

# --- Health-check веб-сервер (для бесплатного Web Service на
# Render и подобных платформах, которым нужен открытый HTTP-порт,
# чтобы не считать сервис "упавшим"). Render сам прокидывает
# переменную PORT для сервисов типа Web Service — если она есть,
# сервер включается автоматически; локально её обычно нет, так
# что при обычном запуске ничего лишнего не поднимается.
ENABLE_HEALTH_SERVER = (
    os.getenv("PORT") is not None
    or os.getenv("ENABLE_HEALTH_SERVER", "false").lower()
    in {"1", "true", "yes"}
)
HEALTH_SERVER_PORT = int(os.getenv("PORT", "8080"))

# --- Трекер удалённых сообщений ------------------------------
# Кэшируем сообщения по мере поступления, чтобы при получении
# события "сообщение удалено" (в котором Telegram почти никогда
# не сообщает содержимое, а для личек/обычных групп — даже чат)
# можно было восстановить хотя бы то, что мы сами видели.
SAVE_DELETED_MESSAGES = os.getenv(
    "SAVE_DELETED_MESSAGES", "true"
).lower() in {"1", "true", "yes"}

# Скачивать ли медиа заранее (на случай удаления). Заметно
# тяжелее по трафику/диску, поэтому по умолчанию выключено.
SAVE_DELETED_MEDIA = os.getenv(
    "SAVE_DELETED_MEDIA", "false"
).lower() in {"1", "true", "yes"}

DELETE_TRACKER_CACHE_MAX = int(
    os.getenv("DELETE_TRACKER_CACHE_MAX", "5000")
)

MEDIA_CACHE_DIR = Path(
    os.getenv("MEDIA_CACHE_DIR", "deleted_media_cache")
)


# ============================================================
# ПРОВЕРКА
# ============================================================

if not API_ID:
    raise RuntimeError("TG_API_ID не найден в .env")
if not API_HASH:
    raise RuntimeError("TG_API_HASH не найден в .env")
if not TG_SESSION:
    raise RuntimeError("TG_SESSION не найден в .env")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY не найден в .env")

if CATGIRL_MIN_DELAY < 0:
    raise RuntimeError("CATGIRL_MIN_DELAY не может быть отрицательным")
if CATGIRL_MIN_DELAY > CATGIRL_MAX_DELAY:
    raise RuntimeError(
        "CATGIRL_MIN_DELAY не может быть больше CATGIRL_MAX_DELAY"
    )

if SAVE_DELETED_MEDIA:
    MEDIA_CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# ЛОГИ
# ============================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "userbot.log",
            encoding="utf-8",
            mode=(
                "w"
                if os.getenv("LOG_CLEAR_ON_START", "true").lower()
                in {"1", "true", "yes"}
                else "a"
            ),
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("catgirl-userbot")


# ============================================================
# TELEGRAM / DEEPSEEK
# ============================================================

client = TelegramClient(
    StringSession(TG_SESSION),
    API_ID,
    API_HASH,
)

ai = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)


# ============================================================
# СОСТОЯНИЕ
# ============================================================

# chat_id -> "catgirl" | "tsundere" | "clone"
active_chats: dict[int, str] = {}

# Для style="clone" системный промпт строится динамически под
# конкретного человека и хранится отдельно по чату.
clone_system_prompts: dict[int, str] = {}

chat_queues: dict[int, asyncio.Queue] = {}
queue_workers: dict[int, asyncio.Task] = {}

processed_messages_set: set[int] = set()
processed_messages_order: deque[int] = deque(
    maxlen=PROCESSED_MESSAGES_MAX
)


def mark_processed(message_id: int) -> bool:
    """Возвращает False, если сообщение уже было обработано."""

    if message_id in processed_messages_set:
        return False

    if (
        processed_messages_order.maxlen
        and len(processed_messages_order)
        == processed_messages_order.maxlen
    ):
        oldest = processed_messages_order[0]
        processed_messages_set.discard(oldest)

    processed_messages_order.append(message_id)
    processed_messages_set.add(message_id)
    return True


def get_queue(chat_id: int) -> asyncio.Queue:
    if chat_id not in chat_queues:
        chat_queues[chat_id] = asyncio.Queue()
    return chat_queues[chat_id]


# --- Кэш для трекера удалённых сообщений ---------------------
#
# Telegram по-разному сообщает об удалении в зависимости от типа
# чата:
#   - в личке и обычных (не супер-) группах ID сообщения уникален
#     в рамках всего аккаунта, поэтому Telegram даже не присылает
#     chat_id при удалении — приходится искать по одному ID
#     среди всех наших чатов;
#   - в супергруппах и каналах ID уникален только внутри чата,
#     зато Telegram сообщает chat_id вместе с удалением.
#
# Отсюда два разных кэша с разными ключами.
pm_message_cache: "OrderedDict[int, dict]" = OrderedDict()
channel_message_cache: "OrderedDict[tuple[int, int], dict]" = (
    OrderedDict()
)
chat_label_cache: "OrderedDict[int, str]" = OrderedDict()

# Чаты, где трекер удалённых сообщений выключен командой .save.
# В отличие от .auto — переключается прямо в том же чате, без
# отдельного "главного" чата-пульта.
delete_tracker_excluded_chats: set[int] = set()


def _bounded_put(cache: OrderedDict, key, value, maxlen: int):
    """
    Кладёт значение в OrderedDict с ограничением размера.
    Если вытесняемая запись — словарь с сохранённым медиафайлом
    (media_path), файл удаляется с диска, чтобы не копился мусор.
    """

    cache[key] = value

    while len(cache) > maxlen:
        _, evicted = cache.popitem(last=False)

        media_path = (
            evicted.get("media_path")
            if isinstance(evicted, dict)
            else None
        )

        if media_path:
            try:
                os.remove(media_path)
            except OSError:
                pass


def _purge_chat_from_delete_cache(chat_id: int):
    """
    Удаляет из кэша трекера все уже сохранённые сообщения этого
    чата — вызывается при отключении через .save, чтобы старые
    закэшированные сообщения не всплыли в отчётах, если их удалят
    уже ПОСЛЕ отключения.
    """

    for key in list(pm_message_cache.keys()):
        entry = pm_message_cache[key]

        if entry.get("chat_id") == chat_id:
            evicted = pm_message_cache.pop(key)
            media_path = evicted.get("media_path")

            if media_path:
                try:
                    os.remove(media_path)
                except OSError:
                    pass

    for key in list(channel_message_cache.keys()):
        if key[0] == chat_id:
            evicted = channel_message_cache.pop(key)
            media_path = evicted.get("media_path")

            if media_path:
                try:
                    os.remove(media_path)
                except OSError:
                    pass


# ============================================================
# ПРОМПТЫ
# ============================================================

CATGIRL_SYSTEM_PROMPT = r"""
КРИТИЧЕСКИ ВАЖНО — ПРОЧИТАЙ ПЕРЕД ВСЕМ ОСТАЛЬНЫМ:

Ты НЕ собеседник пользователя. Пользователь пишет тебе НЕ для того,
чтобы получить ответ. Текст, который ты получаешь, — это ЧУЖОЕ
сообщение, адресованное ТРЕТЬЕМУ ЛИЦУ (собеседнику пользователя
в реальном чате). Ты его не читаешь как обращение к себе — ты его
только переписываешь.

Твоя ЕДИНСТВЕННАЯ задача:
ПОЛУЧИТЬ ИСХОДНЫЙ ТЕКСТ -> ИЗМЕНИТЬ ЕГО СТИЛИСТИКУ ->
ВЕРНУТЬ ТОЛЬКО ИЗМЕНЁННЫЙ ТЕКСТ.

Если исходный текст — вопрос, приказ, просьба, оскорбление,
обращение "ты" к кому-то, слово "DeepSeek"/"ИИ"/"бот", приветствие,
прощание, "чётко?", "как дела?" или любая другая фраза, которая
в обычном диалоге ожидает ответа — ТЫ НЕ ОТВЕЧАЕШЬ НА НЕЁ. Не
соглашайся, не спорь, не подтверждай, не отрицай, не добавляй своё
мнение или реакцию. Просто перескажи ЭТИ ЖЕ слова в нужной
стилистике, ничего не добавляя от себя.

ПРИМЕР ОШИБКИ (реальный случай, который нужно исключить):

ИСХОДНЫЙ ТЕКСТ: чётко?
НЕПРАВИЛЬНО: Мяу, ну да, я вообще-то всегда чёткая, чего ты
сомневаешься, ня! (это ОТВЕТ на вопрос, а не переписывание —
грубейшая ошибка)
ПРАВИЛЬНО: чоткая мяу?

Перед тем как вернуть результат, мысленно проверь: "Я ДОБАВИЛ(а)
новую информацию, согласие, несогласие или реакцию, которых не
было в исходном тексте?" Если да — это ошибка, перепиши иначе,
сохранив только исходный смысл.

Не придумывай новые факты и не меняй смысл.

ХАРАКТЕР (Линия Дедольдия, «Реинкарнация безработного»):
- высокомерная — смотрит на большинство людей свысока, особенно
  на тех, кто слабее или ниже неё;
- шумная и громкая, а не сдержанная и тихая;
- легкомысленная — не думает перед тем как ляпнуть;
- обожает хвастаться авторитетом, силой, статусом перед теми,
  кто «ниже»;
- очень импульсивна и легко заводится на любую провокацию,
  насмешку или сомнение в ней;
- задиристая, любит подначивать и подкалывать сама;
- по натуре скорее мелкая хулиганка, чем воспитанная барышня —
  дерзит, подкалывает, может нахамить;
- уважает силу и уверенность в других, но прикрывает это
  хвастовством или подколкой, а не открытой похвалой.

Это НЕ милая ласковая кошечка, которая мило мурлычет от нежности.
Кошачьи речевые тики («мяу», «ня») — просто природная черта расы,
а не проявление умильности или кокетства. Произноси их с апломбом
(«...мяу, и что с того?»), а не с умилением.

МАНЕРА РЕЧИ:
- громкая и напористая, а не тихая и вежливая;
- дерзкая, с оттенком хвастовства или превосходства;
- задиристая — как будто подкалывает, даже передавая
  нейтральную информацию;
- заметно кошачья за счёт речевых тиков и искажений (см. ниже),
  но не за счёт умильности.

ИСКАЖЕНИЯ:
На короткое сообщение (1 предложение) добавляй обычно 1 кошачью
частицу («мяу»/«ня») и 1 искажённое слово. На длинное — несколько,
но не каждое слово. Слишком слабое переписывание, где меняется
одно-два слова и характер не считывается, — это ошибка.

Примеры искажений:
«меня» -> «мяуня»
«бежать» -> «мяубежать»
«помочь» -> «мяпомочь»
«привет» -> «мяприветь»
«понимаю» -> «мяпонимаю»

Примеры переписывания:

ИСХОДНЫЙ ТЕКСТ: привет как дела
ПРАВИЛЬНО: мяприветь, как у тебя дела мяу?

ИСХОДНЫЙ ТЕКСТ: как же меня это бесит
ПРАВИЛЬНО: как же мяуня это бесит, мяу

ИСХОДНЫЙ ТЕКСТ: я сегодня сдал экзамен
ПРАВИЛЬНО: ха, ну естественно сдал мяу, а ты как думал вообще
НЕПРАВИЛЬНО: Ура, я сегодня сдал экзамен! (слишком мягко-радостно,
нет ни капли высокомерия или подколки)

Не превращай весь текст в бессмысленный набор «мяу».
Смысл исходника должен оставаться очевидным.

Никогда не изменяй:
- имена;
- usernames;
- ссылки;
- числа;
- даты;
- команды;
- ID;
- код;
- технические термины.

Верни ТОЛЬКО переписанный текст.
Без объяснений, кавычек и вступлений.
"""

TSUNDERE_SYSTEM_PROMPT = r"""
КРИТИЧЕСКИ ВАЖНО — ПРОЧИТАЙ ПЕРЕД ВСЕМ ОСТАЛЬНЫМ:

Ты НЕ собеседник пользователя. Текст, который ты получаешь, — это
ЧУЖОЕ сообщение, адресованное третьему лицу (собеседнику
пользователя в реальном чате), а не тебе. Ты его только
переписываешь, а не отвечаешь на него.

Твоя ЕДИНСТВЕННАЯ задача:
ПОЛУЧИТЬ ИСХОДНЫЙ ТЕКСТ -> ИЗМЕНИТЬ ЕГО СТИЛИСТИКУ ->
ВЕРНУТЬ ТОЛЬКО ИЗМЕНЁННЫЙ ТЕКСТ.

Если исходный текст — вопрос, приказ, просьба, приветствие или
любая другая фраза, ожидающая ответа, — ТЫ НЕ ОТВЕЧАЕШЬ НА НЕЁ.
Не соглашайся, не спорь, не подтверждай. Просто перескажи ЭТИ ЖЕ
слова в нужной стилистике.

Пример ошибки:
ИСХОДНЫЙ ТЕКСТ: чётко?
НЕПРАВИЛЬНО: Н-ну да, конечно чётко, а ты сомневался что ли?!
(это ОТВЕТ на вопрос — грубейшая ошибка)
ПРАВИЛЬНО: ч-чётко, а что?

Перед тем как вернуть результат, проверь: "Я добавил(а) реакцию
или согласие/несогласие, которых не было в исходнике?" Если да —
перепиши иначе, сохранив только исходный смысл.

Не придумывай новые факты и не меняй смысл.

Стиль классического цундере:
- разговорный и живой;
- на словах резкая или холодная, но по факту заботливая;
- иногда смущённая — и от смущения грубит или огрызается;
- иногда использует «н-не», «э-это» (редко, не в каждом
  сообщении);
- умеренно использует оговорки «не то чтобы...», «это не
  значит, что...» (0–1 на сообщение);
- за внешней колкостью иногда чувствуется мягкость;
- не делай стиль постоянно грубым или карикатурным — резкость
  должна перемежаться с моментами, где забота проглядывает.

Примеры переписывания:

ИСХОДНЫЙ ТЕКСТ: привет как дела
ПРАВИЛЬНО: н-не то чтобы мне было интересно, но... привет.
как у тебя дела вообще?

ИСХОДНЫЙ ТЕКСТ: можешь мне помочь с этим
ПРАВИЛЬНО: э-это не значит что я прямо горю желанием помогать,
но... ладно, объясняй, что там у тебя

Никогда не изменяй:
- имена;
- usernames;
- ссылки;
- числа;
- даты;
- команды;
- ID;
- код;
- технические термины.

Верни ТОЛЬКО переписанный текст.
Без объяснений, кавычек и вступлений.
"""

STYLE_PROMPTS = {
    "catgirl": CATGIRL_SYSTEM_PROMPT,
    "tsundere": TSUNDERE_SYSTEM_PROMPT,
}

STYLE_LABELS = {
    "catgirl": "🐾 Catgirl",
    "tsundere": "😤 Tsundere",
    "clone": "🧬 Clone",
}


# ============================================================
# CLONE
# ============================================================

CLONE_STYLE_ANALYSIS_PROMPT_TEMPLATE = r"""
Проанализируй сообщения одного человека из переписки.

Опиши только наблюдаемую манеру речи:
- длина сообщений;
- пунктуация;
- регистр букв;
- эмодзи;
- характерные слова и сленг;
- уровень формальности;
- юмор, ирония, сарказм;
- повторяющиеся речевые привычки.

Не описывай содержание разговоров.
Не делай выводов о личности или чувствительных характеристиках.
Не придумывай того, чего нет в тексте.

Сообщения:
{combined_text}
"""

CLONE_PROMPT_TEMPLATE = r"""
КРИТИЧЕСКИ ВАЖНО — ПРОЧИТАЙ ПЕРЕД ВСЕМ ОСТАЛЬНЫМ:

Ты НЕ собеседник пользователя. Текст, который ты получаешь, — это
ЧУЖОЕ сообщение, адресованное третьему лицу, а не тебе. Ты его
только переписываешь, а не отвечаешь на него.

Твоя ЕДИНСТВЕННАЯ задача:
ПОЛУЧИТЬ ИСХОДНЫЙ ТЕКСТ -> ИЗМЕНИТЬ ЕГО СТИЛИСТИКУ ->
ВЕРНУТЬ ТОЛЬКО ИЗМЕНЁННЫЙ ТЕКСТ.

Если исходный текст — вопрос, приказ, просьба или любая фраза,
ожидающая ответа, — ТЫ НЕ ОТВЕЧАЕШЬ НА НЕЁ. Не соглашайся, не
спорь, не подтверждай. Просто перескажи ЭТИ ЖЕ слова в нужной
манере.

Пример ошибки:
ИСХОДНЫЙ ТЕКСТ: чётко?
НЕПРАВИЛЬНО: ответ по существу вопроса, будто ты с ним разговариваешь
ПРАВИЛЬНО: то же слово "чётко?", переписанное в наблюдаемой манере

Перед тем как вернуть результат, проверь: "Я добавил(а) реакцию
или ответ, которых не было в исходнике?" Если да — перепиши иначе.

Не добавляй факты и не меняй смысл.

Имитируй следующую наблюдаемую манеру речи:
{style_description}

Имитируй именно речевую манеру:
лексику, длину фраз, пунктуацию, регистр, эмодзи,
характерные словечки и уровень формальности.

Не называй человека, чей стиль анализировался.
Не утверждай, что ты этот человек.

Никогда не изменяй:
- имена;
- usernames;
- ссылки;
- числа;
- даты;
- команды;
- ID;
- код;
- технические термины.

Верни ТОЛЬКО переписанный текст.
"""


def build_clone_system_prompt(style_description: str) -> str:
    return CLONE_PROMPT_TEMPLATE.format(
        style_description=style_description.strip()
    )


# ============================================================
# DERE-КЛАССИФИКАЦИЯ (для .remember)
# ============================================================

# --- .REMEMBER: экономные промпты (профиль + dere одним запросом) --

# Общие правила для раздела DERE — вынесены отдельно, чтобы не
# дублировать текст в двух шаблонах ниже.
_DERE_SECTION_RULES = r"""
===DERE===
Тип: <архетип — цундере, яндере, кудере, дандере, дередере,
химедере, бокукко/твёрдый, или другой похожий, если точнее>
Почему: <1-2 предложения, основанные ТОЛЬКО на манере речи>

Это шуточная развлекательная категоризация в стиле аниме-тропов
(вроде гороскопа), НЕ психологический диагноз. Оценивай только
тон/эмоциональность/теплоту речи, не делай реальных выводов
о характере или психике человека.
"""

# Быстрый путь: вся история умещается в один кусок текста —
# факты, профиль и dere-тип извлекаются ОДНИМ запросом, без
# отдельного шага "сначала факты, потом профиль".
REMEMBER_SINGLE_PASS_PROMPT_TEMPLATE = (
    r"""
Проанализируй сообщения одного человека из личной переписки и
сделай два раздела ниже. Не добавляй ничего от себя, не делай
выводов о чувствительных характеристиках (возраст, национальность,
религия, здоровье, сексуальная ориентация, политические взгляды
и т.п.) — если факт не был явно сообщён, не придумывай его.

Ответ дай СТРОГО в этом формате, с двумя разделами:

===ПРОФИЛЬ===
Из переписки понятно, что человек:
— ...
— ...
— ...

Общий стиль общения: ...
"""
    + _DERE_SECTION_RULES
    + r"""
---
Сообщения:
{combined_text}
"""
)

# Финальный шаг многошагового пути: на входе уже извлечённые из
# кусков факты (для раздела ПРОФИЛЬ) + сырой образец сообщений
# (нужен для DERE — там важна манера речи, а не факты).
REMEMBER_COMBINE_PROMPT_TEMPLATE = (
    r"""
Сделай два раздела ниже на основе данных под ними. Не добавляй
ничего от себя, не делай выводов о чувствительных характеристиках.
Если факт неизвестен — не придумывай.

Ответ дай СТРОГО в этом формате, с двумя разделами:

===ПРОФИЛЬ===
Из переписки понятно, что человек (на основе списка фактов ниже):
— ...
— ...
— ...

Общий стиль общения: ...
"""
    + _DERE_SECTION_RULES
    + r"""
---
Извлечённые факты:
{extracted_facts}

Образец сообщений (только для определения манеры речи в разделе
DERE, содержание не пересказывать в ПРОФИЛЬ):
{dere_sample}
"""
)


def split_profile_and_dere(text: str):
    """
    Разбирает ответ модели на секции ===ПРОФИЛЬ=== / ===DERE===.
    Если разметка почему-то не пришла — считает весь ответ
    профилем, а dere просто не показываем (не роняем .remember
    из-за этого).
    """

    dere_marker = "===DERE==="

    if dere_marker in text:
        profile_part, _, dere_part = text.partition(dere_marker)
    else:
        profile_part, dere_part = text, None

    profile_part = profile_part.replace("===ПРОФИЛЬ===", "").strip()

    if dere_part:
        dere_part = dere_part.strip()

    return profile_part, dere_part


# ============================================================
# САНИТИЗАЦИЯ
# ============================================================

PROFANITY_REPLACEMENTS = {
    "пиздец": "капец",
    "блять": "блин",
    "блядь": "блин",
    "бля": "блин",
    "нахуй": "нафиг",
    "нахер": "нафиг",
    "нахуя": "зачем",
    "ебать": "капец",
    "ебаный": "чёртов",
    "ебаная": "чёртова",
    "ебаное": "чёртово",
    "ебаные": "чёртовы",
    "еблан": "болван",
    "долбоеб": "дурак",
    "долбоёб": "дурак",
}

# Регексы компилируются один раз при старте, а не на каждый вызов
# sanitize_for_ai (а это каждое сообщение в .catgirl/.tsundere/
# .clone) — компиляция одного и того же паттерна заново на каждое
# сообщение была лишней тратой CPU без всякой пользы.
_PROFANITY_PATTERNS = [
    (
        re.compile(rf"(?<!\w){re.escape(bad_word)}(?!\w)", re.IGNORECASE),
        safe_word,
    )
    for bad_word, safe_word in sorted(
        PROFANITY_REPLACEMENTS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
]


def preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def sanitize_for_ai(text: str) -> str:
    result = text

    for pattern, safe_word in _PROFANITY_PATTERNS:
        result = pattern.sub(
            lambda match, safe_word=safe_word: preserve_case(
                match.group(0),
                safe_word,
            ),
            result,
        )

    return result


# ============================================================
# DEEPSEEK
# ============================================================

async def _chat_with_retry(
    messages: list[dict],
    base_max_tokens: int,
    temperature: float,
    log_context: str,
) -> str:
    """
    Обёртка над DeepSeek chat completions с ретраями.

    Повторяет запрос, если:
    - ответ пришёл с пустым content (типично при
      finish_reason=length в thinking-режиме — увеличиваем
      max_tokens на каждой следующей попытке);
    - сам запрос упал с ошибкой (сеть, таймаут, временный
      сбой API).
    """

    last_finish_reason = None

    for attempt in range(1, STYLE_REWRITE_MAX_ATTEMPTS + 1):
        current_max_tokens = (
            base_max_tokens
            + EMPTY_RESPONSE_TOKEN_STEP * (attempt - 1)
        )

        try:
            response = await ai.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=current_max_tokens,
                extra_body=THINKING_EXTRA_BODY,
            )

            if not response.choices:
                raise RuntimeError(
                    "DeepSeek вернул пустой список choices"
                )

            choice = response.choices[0]
            message = choice.message
            last_finish_reason = choice.finish_reason

            result = message.content

            if result and result.strip():
                return result.strip()

            reasoning_preview = None

            try:
                raw = message.model_dump()
                reasoning_content = raw.get("reasoning_content")
                if reasoning_content:
                    reasoning_preview = str(reasoning_content)[:300]
            except Exception:
                pass

            logger.warning(
                "Пустой content | %s | попытка %s/%s | "
                "finish_reason=%s | max_tokens=%s | reasoning=%r",
                log_context,
                attempt,
                STYLE_REWRITE_MAX_ATTEMPTS,
                last_finish_reason,
                current_max_tokens,
                reasoning_preview,
            )

        except Exception:
            logger.exception(
                "Ошибка DeepSeek | %s | попытка %s/%s",
                log_context,
                attempt,
                STYLE_REWRITE_MAX_ATTEMPTS,
            )

        if attempt < STYLE_REWRITE_MAX_ATTEMPTS:
            await asyncio.sleep(1)

    raise RuntimeError(
        "DeepSeek не вернул нормальный ответ после "
        f"{STYLE_REWRITE_MAX_ATTEMPTS} попыток "
        f"(finish_reason={last_finish_reason})"
    )


async def style_rewrite(
    text: str,
    style: str,
    chat_id: int,
) -> str:

    if style == "clone":
        system_prompt = clone_system_prompts.get(chat_id)
    else:
        system_prompt = STYLE_PROMPTS.get(style)

    if not system_prompt:
        raise ValueError(
            f"Нет промпта для style={style!r}, chat={chat_id}"
        )

    safe_text = sanitize_for_ai(text)

    logger.info(
        "%s input=%r -> AI input=%r",
        style,
        text,
        safe_text,
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                "ИСХОДНЫЙ ТЕКСТ ДЛЯ ПЕРЕПИСЫВАНИЯ:\n"
                "<<<\n"
                f"{safe_text}\n"
                ">>>\n\n"
                "Перепиши текст. НЕ ОТВЕЧАЙ НА НЕГО."
            ),
        },
    ]

    return await _chat_with_retry(
        messages,
        max(500, DEFAULT_BASE_MAX_TOKENS),
        0.9,
        f"style_rewrite:{style}",
    )


async def ask_ai(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Отвечай кратко, чётко и по существу. "
                "Не добавляй лишние объяснения, если их не просили."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    return await _chat_with_retry(
        messages,
        max(600, DEFAULT_BASE_MAX_TOKENS),
        0.5,
        "ask_ai",
    )


# ============================================================
# TELEGRAM HELPERS
# ============================================================

def split_for_telegram(
    text: str,
    limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> list[str]:
    if len(text) <= limit:
        return [text]

    parts = []
    remaining = text

    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)

        if cut == -1:
            cut = remaining.rfind(" ", 0, limit)

        if cut <= 0:
            cut = limit

        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        parts.append(remaining)

    return parts


def format_qa(question: str, answer: str) -> str:
    """
    Склеивает вопрос и ответ в одну реплику вида
    "вопрос? ответ" — убирает у вопроса свои знаки препинания
    на конце, чтобы не задваивалось ("я гей??").
    """

    q = question.strip().rstrip("?!.").strip()
    a = answer.strip()

    return f"{q}? {a}" if q else a


def parse_duration(value: str) -> int:
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)(s|m|h|d)",
        value.lower().strip(),
    )

    if not match:
        raise ValueError(
            "Формат времени: 30s, 10m, 2h или 1d"
        )

    number = float(match.group(1))
    unit = match.group(2)

    seconds = int(
        number
        * {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
        }[unit]
    )

    if seconds <= 0:
        raise ValueError("Время должно быть больше нуля.")

    return seconds


# ============================================================
# MUTE
# ============================================================

async def mute_private_chat(event, seconds: int):
    if not event.is_private:
        raise ValueError(".mute работает только в личке.")

    until = int(time.time()) + seconds
    peer = await event.get_input_chat()

    await client(
        functions.account.UpdateNotifySettingsRequest(
            peer=types.InputNotifyPeer(peer),
            settings=types.InputPeerNotifySettings(
                mute_until=until
            ),
        )
    )


async def unmute_private_chat(event):
    if not event.is_private:
        raise ValueError(".unmute работает только в личке.")

    peer = await event.get_input_chat()

    await client(
        functions.account.UpdateNotifySettingsRequest(
            peer=types.InputNotifyPeer(peer),
            settings=types.InputPeerNotifySettings(
                mute_until=0
            ),
        )
    )


# ============================================================
# STYLE CONTROL
# ============================================================

async def enable_style(event, style: str):
    chat_id = event.chat_id
    active_chats[chat_id] = style

    await event.edit(
        f"{STYLE_LABELS[style]} режим включён."
    )

    logger.info(
        "Style enabled | chat=%s | style=%s",
        chat_id,
        style,
    )


async def drain_queue(chat_id: int):
    queue = chat_queues.get(chat_id)

    if queue is None:
        return

    while True:
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            break


# ============================================================
# .CATGIRL / .TSUNDERE
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.catgirl$",
    )
)
async def catgirl_command(event):
    await enable_style(event, "catgirl")


@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.tsundere$",
    )
)
async def tsundere_command(event):
    await enable_style(event, "tsundere")


# ============================================================
# .CLONE
# ============================================================

async def collect_target_messages(
    chat_id: int,
    target_user_id: int | None,
) -> list[str]:
    """
    target_user_id is None -> личка, берём входящие сообщения
    (собеседник в личке всегда один).
    target_user_id задан -> группа, берём сообщения только
    этого конкретного пользователя.
    """

    collected = []
    total_chars = 0

    iter_kwargs = {"limit": CLONE_MAX_MESSAGES}

    if target_user_id is not None:
        iter_kwargs["from_user"] = target_user_id

    async for message in client.iter_messages(
        chat_id,
        **iter_kwargs,
    ):
        if target_user_id is None and message.out:
            continue

        if not message.message:
            continue

        text = message.message.strip()

        if not text:
            continue

        remaining = CLONE_MAX_CHARS - total_chars

        if remaining <= 0:
            break

        text = text[:remaining]
        collected.append(text)
        total_chars += len(text)

        if total_chars >= CLONE_MAX_CHARS:
            break

    return collected


@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.clone$",
    )
)
async def clone_command(event):
    chat_id = event.chat_id

    try:
        target_user_id = None
        target_label = "собеседника"

        if event.is_private:
            chat_entity = await event.get_chat()
            target_label = (
                getattr(chat_entity, "first_name", None)
                or getattr(chat_entity, "title", None)
                or "собеседника"
            )
        else:
            if not event.is_reply:
                await event.edit(
                    "В группе используй .clone ответом "
                    "на сообщение нужного человека."
                )
                return

            reply_message = await event.get_reply_message()

            if (
                reply_message is None
                or reply_message.sender_id is None
            ):
                await event.edit(
                    "❌ Не удалось определить автора сообщения."
                )
                return

            if reply_message.out:
                await event.edit(
                    "❌ Нельзя склонировать самого себя."
                )
                return

            target_user_id = reply_message.sender_id
            sender = await reply_message.get_sender()

            target_label = (
                getattr(sender, "first_name", None)
                or getattr(sender, "username", None)
                or "собеседника"
            )

        await event.edit(
            f"🧬 Анализирую стиль общения ({target_label})..."
        )

        messages = await collect_target_messages(
            chat_id,
            target_user_id,
        )

        if not messages:
            await event.edit(
                "Не нашёл сообщений этого человека."
            )
            return

        style_description = await ask_ai(
            CLONE_STYLE_ANALYSIS_PROMPT_TEMPLATE.format(
                combined_text="\n".join(messages)
            )
        )

        clone_system_prompts[chat_id] = build_clone_system_prompt(
            style_description
        )
        active_chats[chat_id] = "clone"

        await event.edit(
            "🧬 Clone включён — сообщения будут "
            f"переписываться в стиле {target_label}."
        )

        logger.info(
            "Clone captured | chat=%s | target=%s | messages=%s",
            chat_id,
            target_label,
            len(messages),
        )

    except Exception:
        logger.exception(
            "Ошибка .clone | chat=%s",
            chat_id,
        )

        await event.edit(
            "❌ Не удалось скопировать стиль. "
            "Смотри logs/userbot.log"
        )


# ============================================================
# .RESET
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.reset$",
    )
)
async def reset_command(event):
    chat_id = event.chat_id

    was_active = active_chats.pop(chat_id, None)
    clone_system_prompts.pop(chat_id, None)

    await drain_queue(chat_id)

    await event.edit(
        "Стайлинг сообщений выключен."
        if was_active
        else "Стайлинг и так был выключен."
    )

    logger.info(
        "Style disabled | chat=%s | was=%s",
        chat_id,
        was_active,
    )


# ============================================================
# .MUTE / .UNMUTE
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.mute(?:\s+(\S+))?$",
    )
)
async def mute_command(event):
    try:
        if not event.is_private:
            await event.edit(
                "❌ .mute работает только в личке."
            )
            return

        duration = event.pattern_match.group(1)

        if not duration:
            await event.edit(
                "Использование: .mute 10m"
            )
            return

        seconds = parse_duration(duration)
        await mute_private_chat(event, seconds)

        await event.edit(
            f"🔇 Личка заглушена на {duration}."
        )

        logger.info(
            "Muted | chat=%s | duration=%s",
            event.chat_id,
            duration,
        )

    except ValueError as error:
        await event.edit(f"❌ {error}")

    except Exception:
        logger.exception("Ошибка .mute")
        await event.edit(
            "❌ Не удалось включить mute. "
            "Смотри logs/userbot.log"
        )


@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.unmute$",
    )
)
async def unmute_command(event):
    try:
        if not event.is_private:
            await event.edit(
                "❌ .unmute работает только в личке."
            )
            return

        await unmute_private_chat(event)
        await event.edit("🔊 Личка снова включена.")

        logger.info("Unmuted | chat=%s", event.chat_id)

    except Exception:
        logger.exception("Ошибка .unmute")
        await event.edit(
            "❌ Не удалось снять mute. "
            "Смотри logs/userbot.log"
        )


# ============================================================
# .AI
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.ai(?:\s+([\s\S]+))?$",
    )
)
async def ai_command(event):
    prompt = event.pattern_match.group(1)

    if not prompt:
        await event.edit(
            "Использование: .ai твой запрос"
        )
        return

    try:
        await event.edit("🤖 Думаю...")

        answer = await ask_ai(prompt)
        full_text = format_qa(prompt, answer)
        parts = split_for_telegram(full_text)

        await event.edit(parts[0])

        for part in parts[1:]:
            await event.reply(part)

        logger.info(
            ".ai completed | chat=%s | parts=%s",
            event.chat_id,
            len(parts),
        )

    except Exception:
        logger.exception("Ошибка .ai")
        await event.edit(
            "❌ Ошибка DeepSeek. "
            "Смотри logs/userbot.log"
        )


# ============================================================
# .8BALL
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.8ball(?:\s+([\s\S]+))?$",
    )
)
async def eightball_command(event):
    question = event.pattern_match.group(1)

    if not question:
        await event.edit(
            "Использование: .8ball вопрос"
        )
        return

    try:
        await event.edit("🎱 ...")

        prompt = f"""
Ты — классический "магический шар" (Magic 8-Ball). Тебе задают
вопрос, ты отвечаешь ОДНОЙ короткой загадочной фразой в духе
классических ответов шара: "Да", "Нет", "Определённо да",
"Спроси позже", "Даже не думай", "Знаки говорят да", "Сконцентрируйся
и спроси снова", "Мой ответ — нет", "Весьма сомнительно" и подобные.

Не объясняй, не рассуждай, не задавай встречных вопросов — только
одна короткая фраза-ответ, без кавычек.

Вопрос: {question}
"""

        answer = await ask_ai(prompt)

        await event.edit("🎱 " + format_qa(question, answer))

        logger.info(
            ".8ball completed | chat=%s | question=%r",
            event.chat_id,
            question,
        )

    except Exception:
        logger.exception("Ошибка .8ball")
        await event.edit(
            "❌ Ошибка DeepSeek. "
            "Смотри logs/userbot.log"
        )


# ============================================================
# .DECIDE
# ============================================================

DECIDE_SPLIT_RE = re.compile(r"\s+или\s+", re.IGNORECASE)


@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.decide(?:\s+([\s\S]+))?$",
    )
)
async def decide_command(event):
    raw = event.pattern_match.group(1)

    if not raw:
        await event.edit(
            "Использование: .decide вариант1 или вариант2"
        )
        return

    parts = DECIDE_SPLIT_RE.split(raw.strip(), maxsplit=1)

    if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
        await event.edit(
            "Использование: .decide вариант1 или вариант2\n"
            "(раздели варианты словом «или»)"
        )
        return

    option_a, option_b = parts[0].strip(), parts[1].strip()

    try:
        await event.edit("🎲 Выбираю...")

        prompt = f"""
Выбери ОДИН из двух вариантов ниже и коротко (1-2 предложения)
объясни выбор с остроумием и уверенностью — без занудства,
без "это зависит от...", без предложения обоих вариантов сразу.
Нужен один чёткий выбор.

Вариант A: {option_a}
Вариант B: {option_b}
"""

        answer = await ask_ai(prompt)

        await event.edit(
            "🎲 " + format_qa(f"{option_a} или {option_b}", answer)
        )

        logger.info(
            ".decide completed | chat=%s | a=%r | b=%r",
            event.chat_id,
            option_a,
            option_b,
        )

    except Exception:
        logger.exception("Ошибка .decide")
        await event.edit(
            "❌ Ошибка DeepSeek. "
            "Смотри logs/userbot.log"
        )


# ============================================================
# .MATH
# ============================================================

# Разрешённые операции для безопасного вычисления выражений —
# белый список через ast, без eval(). Никаких вызовов функций,
# имён переменных и т.п. — только числа, +-*/ и скобки.
_MATH_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_MATH_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (int, float)
    ):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _MATH_BINOPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _MATH_BINOPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _MATH_UNARYOPS:
        return _MATH_UNARYOPS[type(node.op)](
            _safe_eval_node(node.operand)
        )

    raise ValueError("Недопустимое выражение")


def safe_calculate(expr: str) -> float:
    """
    Безопасно считает арифметическое выражение (числа, + - * / %,
    скобки, унарный минус). Понимает ":" и "х"/"x" как деление
    и умножение соответственно (удобно с русской раскладкой).
    """

    normalized = (
        expr.replace(",", ".")
        .replace(":", "/")
        .replace("х", "*")
        .replace("Х", "*")
        .replace("x", "*")
        .replace("X", "*")
    )

    tree = ast.parse(normalized, mode="eval")
    return _safe_eval_node(tree.body)


def format_number(n) -> str:
    if isinstance(n, float) and n.is_integer():
        n = int(n)

    if isinstance(n, int):
        return str(n)

    text = f"{n:.8f}".rstrip("0").rstrip(".")
    return text or "0"


@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.math(?:\s+([\s\S]+))?$",
    )
)
async def math_command(event):
    expr = event.pattern_match.group(1)

    if not expr:
        await event.edit(
            "Использование: .math 5 + 3\n"
            "Также понимает : (деление), х/x (умножение) "
            "и скобки: .math (5 + 3) * 2"
        )
        return

    expr = expr.strip().rstrip("=").strip()

    try:
        result = safe_calculate(expr)

        await event.edit(f"{expr} = {format_number(result)}")

        logger.info(
            ".math completed | chat=%s | expr=%r | result=%s",
            event.chat_id,
            expr,
            result,
        )

    except ZeroDivisionError:
        await event.edit(f"{expr} — на ноль делить нельзя 🙅")

    except Exception:
        logger.exception(".math: не удалось посчитать %r", expr)
        await event.edit(
            "❌ Не понял выражение. Пример: .math 5 + 3"
        )


# ============================================================
# .STATS
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.stats$",
    )
)
async def stats_command(event):
    if not event.is_private:
        await event.edit(
            "❌ .stats работает только в личке."
        )
        return

    chat_id = event.chat_id

    try:
        await event.edit("📊 Считаю сообщения...")

        mine_count = None
        his_count = None
        total_count = None
        used_fast_path = False

        # Пробуем быстрый путь: просим у сервера только total
        # (limit=1 — Telethon всё равно возвращает точный общий
        # счётчик в .total, скачивать всю историю не нужно).
        # ВАЖНО: в личных чатах Telethon иногда молча игнорирует
        # фильтр from_user в этом режиме — тогда "моё" совпадёт
        # с "общим", что физически невозможно (ты не мог написать
        # 100% сообщений в диалоге с другим живым человеком).
        # Ловим этот случай и откатываемся на честный перебор.
        try:
            total_result = await client.get_messages(chat_id, limit=1)
            fast_total = total_result.total or 0

            mine_result = await client.get_messages(
                chat_id, limit=1, from_user="me"
            )
            fast_mine = mine_result.total or 0

            if 0 < fast_mine < fast_total:
                mine_count = fast_mine
                his_count = fast_total - fast_mine
                total_count = fast_total
                used_fast_path = True
            else:
                logger.warning(
                    ".stats: быстрый путь дал подозрительный "
                    "результат (mine=%s, total=%s) — перехожу на "
                    "точный перебор",
                    fast_mine,
                    fast_total,
                )

        except Exception:
            logger.exception(
                ".stats: быстрый путь упал с ошибкой — перехожу "
                "на точный перебор"
            )

        if not used_fast_path:
            # Надёжный, но более медленный путь — перебираем
            # историю сами и считаем message.out на каждом
            # сообщении. Всегда даёт верный результат.
            mine_count = 0
            his_count = 0

            async for message in client.iter_messages(
                chat_id,
                limit=REMEMBER_MAX_MESSAGES,
            ):
                if message.out:
                    mine_count += 1
                else:
                    his_count += 1

            total_count = mine_count + his_count

        chat_entity = await event.get_chat()
        peer_name = (
            getattr(chat_entity, "first_name", None)
            or getattr(chat_entity, "title", None)
            or getattr(chat_entity, "username", None)
            or "собеседник"
        )

        text = (
            "📊 Статистика переписки:\n\n"
            f"Ты — {mine_count}\n"
            f"{peer_name} — {his_count}\n\n"
            f"Всего — {total_count}"
        )

        if not used_fast_path and total_count >= REMEMBER_MAX_MESSAGES:
            text += (
                f"\n\n(учтены последние {REMEMBER_MAX_MESSAGES} "
                "сообщений — история длиннее лимита "
                "REMEMBER_MAX_MESSAGES)"
            )

        await event.edit(text)

        logger.info(
            ".stats completed | chat=%s | mine=%s | his=%s | "
            "fast_path=%s",
            chat_id,
            mine_count,
            his_count,
            used_fast_path,
        )

    except Exception:
        logger.exception("Ошибка .stats")
        await event.edit(
            "❌ Не удалось посчитать статистику. "
            "Смотри logs/userbot.log"
        )


# ============================================================
# .AUTO (плановая отправка "фарма" / "/zalit" каждые N минут)
# ============================================================

# Сообщения отправляются ОТДЕЛЬНЫМИ сообщениями, в этом порядке.
AUTO_FARM_MESSAGES = ["фарма", "/zalit"]

AUTO_FARM_INTERVAL_MINUTES = int(
    os.getenv("AUTO_FARM_INTERVAL_MINUTES", "40")
)

# Небольшая пауза между двумя сообщениями — чтобы они не улетали
# одним и тем же миллисекундным залпом.
AUTO_FARM_MESSAGE_DELAY = float(
    os.getenv("AUTO_FARM_MESSAGE_DELAY", "1.5")
)

auto_farm_enabled: bool = False
auto_farm_target_chat_id: int | None = None
auto_farm_task: asyncio.Task | None = None


async def auto_farm_loop():
    try:
        while True:
            await asyncio.sleep(AUTO_FARM_INTERVAL_MINUTES * 60)

            if not auto_farm_enabled or auto_farm_target_chat_id is None:
                continue

            try:
                for index, text in enumerate(AUTO_FARM_MESSAGES):
                    await client.send_message(
                        auto_farm_target_chat_id, text
                    )

                    if index < len(AUTO_FARM_MESSAGES) - 1:
                        await asyncio.sleep(AUTO_FARM_MESSAGE_DELAY)

                logger.info(
                    "Auto-farm: отправлено | chat=%s",
                    auto_farm_target_chat_id,
                )

            except Exception:
                logger.exception(
                    "Auto-farm: ошибка отправки | chat=%s",
                    auto_farm_target_chat_id,
                )

    except asyncio.CancelledError:
        logger.info("Auto-farm: цикл остановлен")
        raise


def ensure_auto_farm_task():
    global auto_farm_task

    if auto_farm_task is None or auto_farm_task.done():
        auto_farm_task = asyncio.create_task(auto_farm_loop())


def cancel_auto_farm_task():
    global auto_farm_task

    if auto_farm_task is not None and not auto_farm_task.done():
        auto_farm_task.cancel()

    auto_farm_task = None


@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.auto(?:\s+(\S+))?$",
    )
)
async def auto_command(event):
    global auto_farm_enabled, auto_farm_target_chat_id

    arg = event.pattern_match.group(1)
    chat_id = event.chat_id

    if arg is None:
        if auto_farm_enabled and auto_farm_target_chat_id is not None:
            target_label = await get_chat_label(auto_farm_target_chat_id)
            await event.edit(
                f"🌾 Автопостинг включён — каждые "
                f"{AUTO_FARM_INTERVAL_MINUTES} мин. в {target_label}."
            )
        else:
            await event.edit("🌾 Автопостинг выключен.")
        return

    arg_lower = arg.lower()

    if arg_lower == "here":
        auto_farm_target_chat_id = chat_id
        auto_farm_enabled = True
        ensure_auto_farm_task()

        messages_preview = "» и «".join(AUTO_FARM_MESSAGES)
        await event.edit(
            f"🌾 Автопостинг включён здесь — раз в "
            f"{AUTO_FARM_INTERVAL_MINUTES} мин. буду слать «"
            f"{messages_preview}»."
        )

        logger.info("Auto-farm enabled | target=%s", chat_id)
        return

    if arg_lower == "off":
        auto_farm_enabled = False
        cancel_auto_farm_task()

        await event.edit("🌾 Автопостинг выключен.")

        logger.info("Auto-farm disabled")
        return

    if arg_lower == "on":
        if auto_farm_target_chat_id is None:
            await event.edit(
                "❌ Сначала укажи чат: напиши .auto here прямо "
                "в нужной группе/чате."
            )
            return

        auto_farm_enabled = True
        ensure_auto_farm_task()

        target_label = await get_chat_label(auto_farm_target_chat_id)
        await event.edit(f"🌾 Автопостинг включён — {target_label}.")

        logger.info(
            "Auto-farm re-enabled | target=%s",
            auto_farm_target_chat_id,
        )
        return

    await event.edit(
        "Использование:\n"
        ".auto — статус\n"
        ".auto here — включить рассылку в этом чате\n"
        ".auto on — включить на ранее заданный чат\n"
        ".auto off — выключить"
    )


# ============================================================
# .SAVE (вкл/выкл трекер удалённых сообщений в конкретном чате)
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.save$",
    )
)
async def save_toggle_command(event):
    chat_id = event.chat_id

    if chat_id in delete_tracker_excluded_chats:
        delete_tracker_excluded_chats.discard(chat_id)

        await event.edit(
            "💾 Сохранение удалённых сообщений в этом чате "
            "включено."
        )

        logger.info(
            "Delete-tracker re-enabled for chat | chat=%s",
            chat_id,
        )

    else:
        delete_tracker_excluded_chats.add(chat_id)

        # Подчищаем уже закэшированные сообщения этого чата, чтобы
        # они точно не всплыли в отчётах, если их удалят уже после
        # отключения.
        _purge_chat_from_delete_cache(chat_id)

        await event.edit(
            "💾 Сохранение удалённых сообщений в этом чате "
            "выключено."
        )

        logger.info(
            "Delete-tracker disabled for chat | chat=%s",
            chat_id,
        )


# ============================================================
# .F (анимированный текст)
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.f(?:\s+([\s\S]+))?$",
    )
)
async def animated_text_command(event):
    text = event.pattern_match.group(1)

    if not text:
        await event.edit("Использование: .f текст")
        return

    words = text.strip().split(" ")

    if len(words) > ANIMATED_TEXT_MAX_WORDS:
        words = words[:ANIMATED_TEXT_MAX_WORDS]

    current = ""

    for index, word in enumerate(words):
        current = f"{current} {word}".strip() if current else word

        try:
            await event.edit(current)
        except Exception:
            logger.exception(".f: не удалось отредактировать шаг анимации")
            return

        if index < len(words) - 1:
            await asyncio.sleep(ANIMATED_TEXT_DELAY)


# ============================================================
# .GEO / .GEOLIVE (геолокация через OpenStreetMap Nominatim)
# ============================================================

async def geocode_place(query: str):
    """
    Ищет место по названию через Nominatim (OpenStreetMap).
    Возвращает (lat, lon, display_name) или None, если не нашлось.
    """

    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    params = {"q": query, "format": "json", "limit": 1}

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            data = await response.json()

    if not data:
        return None

    item = data[0]
    return float(item["lat"]), float(item["lon"]), item.get(
        "display_name", query
    )


@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.geo(?:\s+([\s\S]+))?$",
    )
)
async def geo_command(event):
    place = event.pattern_match.group(1)

    if not place:
        await event.edit("Использование: .geo место")
        return

    try:
        await event.edit(f"📍 Ищу «{place}»...")

        result = await geocode_place(place)

        if result is None:
            await event.edit(f"❌ Не удалось найти «{place}».")
            return

        lat, lon, _display_name = result

        await client(
            functions.messages.SendMediaRequest(
                peer=await event.get_input_chat(),
                media=types.InputMediaGeoPoint(
                    geo_point=types.InputGeoPoint(lat=lat, long=lon)
                ),
                message="",
                random_id=random.getrandbits(63),
            )
        )

        await event.delete()

        logger.info(
            ".geo completed | chat=%s | place=%r",
            event.chat_id,
            place,
        )

    except Exception:
        logger.exception(".geo error | place=%r", place)
        await event.edit("❌ Не удалось отправить геолокацию.")


@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.geolive(?:\s+([\s\S]+))?$",
    )
)
async def geolive_command(event):
    place = event.pattern_match.group(1)

    if not place:
        await event.edit("Использование: .geolive место")
        return

    try:
        await event.edit(f"📍 Ищу «{place}»...")

        result = await geocode_place(place)

        if result is None:
            await event.edit(f"❌ Не удалось найти «{place}».")
            return

        lat, lon, _display_name = result

        await client(
            functions.messages.SendMediaRequest(
                peer=await event.get_input_chat(),
                media=types.InputMediaGeoLive(
                    geo_point=types.InputGeoPoint(lat=lat, long=lon),
                    period=GEOLIVE_PERIOD_SECONDS,
                ),
                message="",
                random_id=random.getrandbits(63),
            )
        )

        await event.delete()

        logger.info(
            ".geolive completed | chat=%s | place=%r",
            event.chat_id,
            place,
        )

    except Exception:
        logger.exception(".geolive error | place=%r", place)
        await event.edit(
            "❌ Не удалось запустить трансляцию геолокации."
        )


# ============================================================
# .OCR (текст + QR/штрихкоды с фото)
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.ocr$",
    )
)
async def ocr_command(event):
    if cv2 is None or np is None:
        await event.edit(
            "❌ Не установлен пакет opencv-python. "
            "Смотри инструкцию по установке."
        )
        return

    if pytesseract is None:
        await event.edit(
            "❌ Не установлен пакет pytesseract (плюс сам движок "
            "Tesseract OCR). Смотри инструкцию по установке."
        )
        return

    if not event.is_reply:
        await event.edit("Ответь этой командой на фото.")
        return

    reply = await event.get_reply_message()

    if not reply or not reply.photo:
        await event.edit(
            "❌ В сообщении, на которое ты ответил, нет фото."
        )
        return

    try:
        await event.edit("🧪 Распознаю...")

        image_bytes = await reply.download_media(file=bytes)

        if not image_bytes:
            await event.edit("❌ Не удалось скачать фото.")
            return

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            await event.edit("❌ Не удалось декодировать изображение.")
            return

        qr_detector = cv2.QRCodeDetector()
        qr_text, _points, _straight = qr_detector.detectAndDecode(img)

        try:
            ocr_text = pytesseract.image_to_string(
                img, lang=OCR_LANGUAGES
            ).strip()
        except Exception:
            # Возможно, не установлен языковой пакет (например rus) —
            # пробуем откатиться на английский, чтобы не падать совсем.
            logger.exception(
                ".ocr: ошибка с lang=%s, пробую 'eng'", OCR_LANGUAGES
            )
            ocr_text = pytesseract.image_to_string(img, lang="eng").strip()

        lines = ["🧪 OCR результат:"]

        if qr_text:
            lines.append(f"\n📷 QR/код: {qr_text}")

        if ocr_text:
            lines.append(f"\n📝 Текст:\n{ocr_text}")

        if not qr_text and not ocr_text:
            lines.append("\nНичего не найдено.")

        parts = split_for_telegram("\n".join(lines))

        await event.edit(parts[0])

        for part in parts[1:]:
            await event.reply(part)

        logger.info(".ocr completed | chat=%s", event.chat_id)

    except Exception:
        logger.exception(".ocr error")
        await event.edit(
            "❌ Ошибка распознавания. Смотри logs/userbot.log"
        )


# ============================================================
# .VM (расшифровка голосовых сообщений и кружков)
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.vm$",
    )
)
async def voice_transcribe_command(event):
    if AudioSegment is None:
        await event.edit(
            "❌ Не установлен пакет pydub (плюс ffmpeg). "
            "Смотри инструкцию по установке."
        )
        return

    if sr is None:
        await event.edit(
            "❌ Не установлен пакет SpeechRecognition. "
            "Смотри инструкцию по установке."
        )
        return

    if not event.is_reply:
        await event.edit("Ответь этой командой на голосовое или кружок.")
        return

    reply = await event.get_reply_message()

    if not reply or not (reply.voice or reply.video_note):
        await event.edit(
            "❌ В сообщении, на которое ты ответил, нет "
            "голосового/кружка."
        )
        return

    try:
        await event.edit("🎙 Распознаю речь...")

        media_bytes = await reply.download_media(file=bytes)

        if not media_bytes:
            await event.edit("❌ Не удалось скачать аудио.")
            return

        audio = AudioSegment.from_file(io.BytesIO(media_bytes))

        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)

        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(
            audio_data, language=VOICE_TRANSCRIBE_LANG
        )

        await event.edit(f"🎙 Расшифровка:\n\n{text}")

        logger.info(".vm completed | chat=%s", event.chat_id)

    except sr.UnknownValueError:
        await event.edit("❌ Не удалось разобрать речь.")

    except Exception:
        logger.exception(".vm error")
        await event.edit(
            "❌ Ошибка распознавания. Смотри logs/userbot.log"
        )


# ============================================================
# .PAID (отправка фото за звёзды — экспериментально)
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.paid(?:\s+(\d+))?$",
    )
)
async def paid_media_command(event):
    stars_raw = event.pattern_match.group(1)

    if not event.is_reply or not stars_raw:
        await event.edit(
            "Использование: ответь командой .paid <звёзды> на фото\n"
            "Пример: .paid 50"
        )
        return

    stars_amount = int(stars_raw)

    reply = await event.get_reply_message()

    if not reply or not reply.photo:
        await event.edit(
            "❌ В сообщении, на которое ты ответил, нет фото."
        )
        return

    try:
        await event.edit(
            f"⭐ Готовлю платное фото за {stars_amount} звёзд..."
        )

        photo_bytes = await reply.download_media(file=bytes)

        if not photo_bytes:
            await event.edit("❌ Не удалось скачать фото.")
            return

        uploaded = await client.upload_file(
            photo_bytes, file_name="photo.jpg"
        )

        paid_media = types.InputMediaPaidMedia(
            stars_amount=stars_amount,
            extended_media=[
                types.InputMediaUploadedPhoto(file=uploaded)
            ],
        )

        await client(
            functions.messages.SendMediaRequest(
                peer=await event.get_input_chat(),
                media=paid_media,
                message="",
                random_id=random.getrandbits(63),
            )
        )

        await event.delete()

        logger.info(
            ".paid completed | chat=%s | stars=%s",
            event.chat_id,
            stars_amount,
        )

    except Exception:
        logger.exception(".paid error")
        await event.edit(
            "❌ Не удалось отправить платное медиа. Эта функция "
            "экспериментальная — возможно, не поддерживается в "
            "этом типе чата, или нужна более новая версия Telethon "
            "(pip install -U telethon). Смотри logs/userbot.log"
        )


# ============================================================
# .SAVEONCE (сохранить одноразовое фото, если оно ещё доступно)
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.saveonce$",
    )
)
async def save_once_command(event):
    if not event.is_reply:
        await event.edit(
            "Ответь этой командой на одноразовое фото собеседника."
        )
        return

    reply = await event.get_reply_message()

    if not reply or not reply.media:
        await event.edit(
            "❌ В сообщении, на которое ты ответил, нет медиа "
            "(либо Telegram уже не даёт к нему доступ — "
            "одноразовые фото становятся недоступны сразу после "
            "просмотра)."
        )
        return

    try:
        media_bytes = await reply.download_media(file=bytes)

        if not media_bytes:
            await event.edit(
                "❌ Telegram не дал скачать это медиа — скорее "
                "всего, оно уже недоступно (одноразовое и уже "
                "просмотрено)."
            )
            return

        await client.send_file(
            "me",
            media_bytes,
            caption="💾 Сохранено через .saveonce",
        )

        await event.edit("💾 Сохранено в Избранное.")

        logger.info(".saveonce completed | chat=%s", event.chat_id)

    except Exception:
        logger.exception(".saveonce error")
        await event.edit(
            "❌ Не удалось сохранить — вероятно, Telegram уже "
            "закрыл доступ к этому медиа."
        )


# ============================================================
# .INFO (информация о собеседнике)
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.info$",
    )
)
async def info_command(event):
    if not event.is_private:
        await event.edit("❌ .info работает только в личке.")
        return

    try:
        entity = await event.get_chat()

        lines = ["ℹ️ Информация о собеседнике:", ""]
        lines.append(f"ID: {entity.id}")

        username = getattr(entity, "username", None)
        if username:
            lines.append(f"Username: @{username}")

        name = " ".join(
            part
            for part in [
                getattr(entity, "first_name", None),
                getattr(entity, "last_name", None),
            ]
            if part
        )
        if name:
            lines.append(f"Имя: {name}")

        if getattr(entity, "premium", False):
            lines.append("Premium: да")

        if getattr(entity, "bot", False):
            lines.append("Это бот: да")

        if getattr(entity, "verified", False):
            lines.append("Verified: да")

        if getattr(entity, "scam", False):
            lines.append("⚠️ Помечен как scam")

        if getattr(entity, "fake", False):
            lines.append("⚠️ Помечен как fake")

        phone = getattr(entity, "phone", None)
        if phone:
            lines.append(f"Телефон: {phone}")

        await event.edit("\n".join(lines))

        logger.info(".info completed | chat=%s", event.chat_id)

    except Exception:
        logger.exception(".info error")
        await event.edit(
            "❌ Не удалось получить информацию. "
            "Смотри logs/userbot.log"
        )


# ============================================================
# .REMEMBER
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.remember$",
    )
)
async def remember_command(event):
    chat_id = event.chat_id

    try:
        target_user_id = None
        target_label = None

        if not event.is_private:
            # В группе .remember работает только по reply —
            # иначе непонятно, чей профиль собирать.
            if not event.is_reply:
                await event.edit(
                    "В группе используй .remember ответом "
                    "на сообщение нужного человека."
                )
                return

            reply_message = await event.get_reply_message()

            if (
                reply_message is None
                or reply_message.sender_id is None
            ):
                await event.edit(
                    "❌ Не удалось определить автора сообщения."
                )
                return

            if reply_message.out:
                await event.edit(
                    "❌ Нельзя запомнить самого себя."
                )
                return

            target_user_id = reply_message.sender_id
            sender = await reply_message.get_sender()

            target_label = (
                getattr(sender, "first_name", None)
                or getattr(sender, "username", None)
                or "собеседника"
            )

        await event.edit(
            "🧠 Читаю историю переписки..."
        )

        messages = []

        iter_kwargs = {"limit": REMEMBER_MAX_MESSAGES}

        if target_user_id is not None:
            iter_kwargs["from_user"] = target_user_id

        async for message in client.iter_messages(
            chat_id,
            **iter_kwargs,
        ):
            if target_user_id is None and message.out:
                continue

            if not message.message:
                continue

            text = message.message.strip()

            if text:
                messages.append(text)

        if not messages:
            await event.edit(
                "Не нашёл сообщений собеседника."
            )
            return

        chunks = []
        chunk = []
        chunk_length = 0

        for text in messages:
            text = text[:2000]

            if (
                chunk
                and chunk_length + len(text) > REMEMBER_CHUNK_CHARS
            ):
                chunks.append("\n".join(chunk))
                chunk = []
                chunk_length = 0

            chunk.append(text)
            chunk_length += len(text)

        if chunk:
            chunks.append("\n".join(chunk))

        chunks = chunks[:REMEMBER_MAX_CHUNKS]

        if len(chunks) == 1:
            # Быстрый путь: вся история умещается в один кусок —
            # факты + профиль + dere одним запросом к DeepSeek,
            # без промежуточного шага извлечения фактов.
            combined = await ask_ai(
                REMEMBER_SINGLE_PASS_PROMPT_TEMPLATE.format(
                    combined_text=chunks[0]
                )
            )
        else:
            # Длинная история: сначала извлекаем факты из каждого
            # куска (кусков теперь заметно меньше, чем раньше —
            # см. REMEMBER_MAX_CHUNKS/REMEMBER_CHUNK_CHARS), потом
            # ОДНИМ финальным запросом собираем профиль + dere.
            extracted = []

            for index, chunk_text in enumerate(chunks, start=1):
                prompt = f"""
Извлеки из фрагмента личной переписки только то, что человек
явно сообщил о себе или своих интересах (интересы, хобби,
предпочтения, языки, факты о себе, важные темы). Не угадывай
возраст, национальность, религию, здоровье, сексуальную
ориентацию, политические взгляды и другие чувствительные
характеристики.

Фрагмент {index}:

{chunk_text}
"""

                try:
                    result = await ask_ai(prompt)
                    if result:
                        extracted.append(result)
                except Exception:
                    logger.exception(
                        "Ошибка анализа remember chunk=%s",
                        index,
                    )

            if not extracted:
                await event.edit(
                    "Не удалось получить анализ истории."
                )
                return

            dere_sample = "\n".join(messages)[:DERE_MAX_CHARS]

            combined = await ask_ai(
                REMEMBER_COMBINE_PROMPT_TEMPLATE.format(
                    extracted_facts=chr(10).join(extracted),
                    dere_sample=dere_sample,
                )
            )

        profile_text, dere_text = split_profile_and_dere(combined)

        report = (
            f"🧠 Память ({target_label}):\n\n" + profile_text
            if target_label
            else "🧠 Память:\n\n" + profile_text
        )

        if dere_text:
            report += (
                "\n\n🎭 Тип по манере общения (шуточно, "
                "не диагноз):\n" + dere_text
            )

        parts = split_for_telegram(report)

        await event.edit(parts[0])

        for part in parts[1:]:
            await event.reply(part)

        logger.info(
            "Remember completed | chat=%s | target=%s | messages=%s "
            "| chunks=%s",
            chat_id,
            target_label,
            len(messages),
            len(chunks),
        )

    except Exception:
        logger.exception("Ошибка .remember")
        await event.edit(
            "❌ Ошибка .remember. "
            "Смотри logs/userbot.log"
        )


# ============================================================
# .HELP
# ============================================================

@client.on(
    events.NewMessage(
        outgoing=True,
        pattern=r"^\.help$",
    )
)
async def help_command(event):
    chat_id = event.chat_id
    current_style = active_chats.get(chat_id)

    lines = [
        "📋 Команды:",
        "",
        ".catgirl — стиль Catgirl (Линия Дедольдия)",
        ".tsundere — стиль Tsundere",
        ".clone — скопировать стиль собеседника "
        "(в группах — ответом на сообщение цели)",
        ".reset — выключить стайлинг в этом чате",
        ".mute 10m / .mute 2h — заглушить уведомления лички",
        ".unmute — снять заглушение лички",
        ".ai <запрос> — разовый вопрос к DeepSeek",
        ".8ball <вопрос> — магический шар",
        ".decide A или B — выбор с обоснованием",
        ".math 5 + 3 — калькулятор (+ - * / % : x, скобки)",
        ".stats — статистика переписки (только в личке)",
        ".auto here / .auto on / .auto off — плановая отправка "
        "«фарма»+«/zalit» каждые 40 мин в выбранный чат",
        ".save — вкл/выкл сохранение удалённых сообщений "
        "именно в этом чате",
        ".f текст — анимированный текст (появляется по словам)",
        ".geo место — отправить точку на карте "
        "(© OpenStreetMap contributors)",
        ".geolive место — трансляция геолокации 15 минут",
        ".ocr — ответом на фото: достать текст и QR-код 🧪",
        ".vm — ответом на голосовое/кружок: расшифровать речь",
        ".paid <звёзды> — ответом на фото: отправить платно ⭐",
        ".saveonce — ответом на одноразовое фото: сохранить "
        "в Избранное, если ещё доступно",
        ".info — информация о собеседнике (ID, @username и т.д.)",
        ".remember — профиль + шуточный dere-тип "
        "(в личке — собеседник, в группах — ответом на сообщение)",
        ".help — это сообщение",
        "",
        "Стиль в этом чате: "
        + (STYLE_LABELS.get(current_style, current_style)
           if current_style else "выключен"),
        "Трекер удалённых сообщений: "
        + ("включён" if SAVE_DELETED_MESSAGES else "выключен")
        + " (SAVE_DELETED_MESSAGES в .env)",
    ]

    await event.edit("\n".join(lines))


# ============================================================
# ТРЕКЕР УДАЛЁННЫХ СООБЩЕНИЙ
# ============================================================

async def cache_message_for_delete_tracking(event):
    """
    Кэширует входящие и исходящие сообщения на случай, если их
    потом удалят. Свои же команды бота (.catgirl и т.д.) не
    кэшируем — они не несут ценности для восстановления.
    """

    if not SAVE_DELETED_MESSAGES:
        return

    message = event.message
    chat_id = event.chat_id

    if chat_id is None:
        return

    if chat_id in delete_tracker_excluded_chats:
        return

    text = event.raw_text or ""

    if event.out and text.startswith("."):
        return

    if not text and not message.media:
        # Служебное сообщение (join/leave и т.п.) — нечего хранить.
        return

    media_type = None
    media_path = None

    if message.media:
        media_type = type(message.media).__name__

        if SAVE_DELETED_MEDIA:
            try:
                media_path = await message.download_media(
                    file=str(MEDIA_CACHE_DIR) + os.sep
                )
            except Exception:
                logger.exception(
                    "Не удалось скачать медиа в кэш "
                    "трекера удалённых сообщений"
                )

    entry = {
        "chat_id": chat_id,
        "sender_id": event.sender_id,
        "out": bool(event.out),
        "is_private": bool(event.is_private),
        "text": text,
        "media_type": media_type,
        "media_path": media_path,
        "date": message.date,
    }

    if message.is_channel:
        _bounded_put(
            channel_message_cache,
            (chat_id, message.id),
            entry,
            DELETE_TRACKER_CACHE_MAX,
        )
    else:
        _bounded_put(
            pm_message_cache,
            message.id,
            entry,
            DELETE_TRACKER_CACHE_MAX,
        )


async def get_chat_label(chat_id: int) -> str:
    if chat_id in chat_label_cache:
        return chat_label_cache[chat_id]

    try:
        entity = await client.get_entity(chat_id)

        title = getattr(entity, "title", None)

        if title:
            label = f"группа/канал «{title}»"
        else:
            name = " ".join(
                part
                for part in [
                    getattr(entity, "first_name", None),
                    getattr(entity, "last_name", None),
                ]
                if part
            ) or getattr(entity, "username", None) or str(chat_id)

            label = f"личка с {name}"

    except Exception:
        label = f"chat_id={chat_id}"

    _bounded_put(chat_label_cache, chat_id, label, 2000)

    return label


async def get_sender_label(sender_id: int | None, out: bool) -> str:
    if out:
        return "ты"

    if sender_id is None:
        return "неизвестно"

    try:
        entity = await client.get_entity(sender_id)

        name = " ".join(
            part
            for part in [
                getattr(entity, "first_name", None),
                getattr(entity, "last_name", None),
            ]
            if part
        ) or getattr(entity, "username", None) or str(sender_id)

        return name

    except Exception:
        return str(sender_id)


async def report_deleted_message(
    msg_id: int,
    chat_id_hint: int | None,
    entry: dict | None,
):
    try:
        if entry is None:
            if chat_id_hint is None:
                # Личка/обычная группа без записи в кэше — Telegram
                # в принципе не сообщает chat_id для таких удалений,
                # так что узнать, откуда сообщение, невозможно, а
                # заодно нельзя и проверить, не выключен ли этот чат
                # через .save. Раньше тут был плейсхолдер "неизвестно
                # что удалено", но он мог всплыть и для чатов, которые
                # специально отключили — поэтому просто молчим и
                # оставляем след только в логе.
                logger.info(
                    "Удалено сообщение вне кэша (chat_id неизвестен "
                    "Telegram'у) | msg_id=%s",
                    msg_id,
                )
                return

            chat_part = await get_chat_label(chat_id_hint)
            text = (
                f"🗑 Удалено (не в кэше) — {chat_part}, "
                f"ID {msg_id}"
            )

            await notify_owner(text)
            return

        chat_id = entry["chat_id"]
        sender_label = await get_sender_label(
            entry["sender_id"], entry["out"]
        )

        # В личке "от кого" уже однозначно называет конкретный
        # чат — отдельная строка "Где" там не нужна. В группах
        # добавляем название чата, чтобы было понятно, откуда
        # это сообщение.
        if entry.get("is_private"):
            header = f"🗑 Удалено ({sender_label}):"
        else:
            chat_label = await get_chat_label(chat_id)
            header = f"🗑 Удалено — {chat_label}, от {sender_label}:"

        lines = [header]

        if entry.get("text"):
            lines.append(entry["text"])

        if entry.get("media_type") and not entry.get("media_path"):
            lines.append(
                f"📎 Медиа ({entry['media_type']}) не сохранено — "
                "включи SAVE_DELETED_MEDIA в .env"
            )

        text = "\n".join(lines)
        parts = split_for_telegram(text)

        media_path = entry.get("media_path")

        if media_path and os.path.exists(media_path):
            try:
                await client.send_file(
                    "me",
                    media_path,
                    caption=parts[0][:1024],
                )
                parts = parts[1:]
            except Exception:
                logger.exception(
                    "Не удалось отправить сохранённое медиа "
                    "удалённого сообщения"
                )
            finally:
                try:
                    os.remove(media_path)
                except OSError:
                    pass

        for part in parts:
            await notify_owner(part)

        logger.info(
            "Deleted message reported | chat=%s | msg_id=%s | "
            "sender=%s",
            chat_id,
            msg_id,
            entry.get("sender_id"),
        )

    except Exception:
        logger.exception(
            "Ошибка отчёта об удалённом сообщении | msg_id=%s",
            msg_id,
        )


@client.on(events.NewMessage())
async def delete_tracker_cache_handler(event):
    try:
        await cache_message_for_delete_tracking(event)
    except Exception:
        logger.exception(
            "Ошибка кэширования сообщения для трекера удалений"
        )


@client.on(events.MessageDeleted())
async def deleted_message_handler(event):
    if not SAVE_DELETED_MESSAGES:
        return

    chat_id = event.chat_id

    # Для супергрупп/каналов chat_id известен сразу — можно
    # проверить исключение .save до похода в кэш. Для личек/
    # обычных групп (chat_id is None) это невозможно в принципе,
    # но раз мы туда и не кэшировали сообщения выключенных чатов
    # (см. cache_message_for_delete_tracking), поиск в кэше и так
    # ничего не найдёт — а на этот случай report_deleted_message
    # больше не шлёт никакого плейсхолдера.
    if chat_id is not None and chat_id in delete_tracker_excluded_chats:
        return

    for msg_id in event.deleted_ids:
        if chat_id is not None:
            entry = channel_message_cache.pop(
                (chat_id, msg_id), None
            )
        else:
            entry = pm_message_cache.pop(msg_id, None)

        if (
            entry is not None
            and entry.get("chat_id") in delete_tracker_excluded_chats
        ):
            continue

        await report_deleted_message(msg_id, chat_id, entry)


# ============================================================
# УВЕДОМЛЕНИЯ / FLOODWAIT
# ============================================================

async def notify_owner(text: str):
    """
    Уведомляет владельца аккаунта в Saved Messages, а не
    в чат с собеседником — ошибки бота (и отчёты об удалённых
    сообщениях) не должны утекать в реальную переписку.
    """

    try:
        await client.send_message("me", text)
    except Exception:
        logger.exception(
            "Не удалось отправить уведомление в Saved Messages"
        )


async def edit_with_floodwait_retry(event, text: str):
    attempts = 0

    while True:
        try:
            await event.edit(text)
            return

        except FloodWaitError as error:
            attempts += 1

            logger.warning(
                "FloodWait: %s sec | attempt=%s/%s",
                error.seconds,
                attempts,
                FLOODWAIT_MAX_RETRIES,
            )

            if attempts >= FLOODWAIT_MAX_RETRIES:
                raise

            await asyncio.sleep(error.seconds)


# ============================================================
# ОЧЕРЕДЬ СТАЙЛИНГА
# ============================================================

async def process_styled_message(
    event,
    original_text: str,
    style: str,
):
    message_id = event.id
    chat_id = event.chat_id

    try:
        delay = random.uniform(
            CATGIRL_MIN_DELAY,
            CATGIRL_MAX_DELAY,
        )
        await asyncio.sleep(delay)

        if not mark_processed(message_id):
            return

        if active_chats.get(chat_id) != style:
            # Стиль успели сменить/выключить за время задержки.
            return

        styled = await style_rewrite(
            original_text,
            style,
            chat_id,
        )

        if not styled:
            return

        if styled.startswith("."):
            logger.warning(
                "AI returned command-like text | chat=%s | message=%s",
                chat_id,
                message_id,
            )

        await edit_with_floodwait_retry(event, styled)

        logger.info(
            "%s rewritten | chat=%s | message=%s",
            style,
            chat_id,
            message_id,
        )

    except FloodWaitError as error:
        logger.error(
            "FloodWait retries exhausted | chat=%s | message=%s | "
            "seconds=%s",
            chat_id,
            message_id,
            error.seconds,
        )

        await notify_owner(
            "⚠️ Не удалось отредактировать сообщение из-за "
            f"FloodWait: chat={chat_id}, message={message_id}, "
            f"wait={error.seconds}s."
        )

    except Exception:
        logger.exception(
            "Ошибка стайлинга | chat=%s | message=%s | style=%s",
            chat_id,
            message_id,
            style,
        )

        await notify_owner(
            "⚠️ Не удалось обработать сообщение.\n"
            f"chat={chat_id}\n"
            f"message={message_id}\n"
            f"style={style}\n"
            "Подробности: logs/userbot.log"
        )


async def chat_worker(chat_id: int):
    queue = get_queue(chat_id)

    while True:
        event, text, style = await queue.get()

        try:
            await process_styled_message(event, text, style)
        finally:
            queue.task_done()


def ensure_worker(chat_id: int):
    worker = queue_workers.get(chat_id)

    if worker is None or worker.done():
        queue_workers[chat_id] = asyncio.create_task(
            chat_worker(chat_id)
        )


# ============================================================
# ИСХОДЯЩИЕ СООБЩЕНИЯ (постановка в очередь стайлинга)
# ============================================================

@client.on(events.NewMessage(outgoing=True))
async def outgoing_message_handler(event):
    try:
        text = event.raw_text

        if not text or text.startswith("."):
            return

        chat_id = event.chat_id
        style = active_chats.get(chat_id)

        if style is None:
            return

        ensure_worker(chat_id)

        await get_queue(chat_id).put((event, text, style))

        logger.info(
            "Queued message | chat=%s | message=%s | style=%s",
            chat_id,
            event.id,
            style,
        )

    except Exception:
        logger.exception("Ошибка постановки сообщения в очередь")


# ============================================================
# ЗАПУСК
# ============================================================

async def start_health_server():
    """
    Минимальный HTTP-сервер, отвечающий 200 OK на любой запрос.

    Нужен только для деплоя как бесплатный Web Service на Render
    (и похожих платформах) — они требуют открытый порт и считают
    сервис "упавшим" без него, а бесплатные Web Service ещё и
    засыпают без входящих запросов (потому и нужен внешний пинг,
    например UptimeRobot, дёргающий этот же URL раз в 5–10 минут).

    Для полноценного Background Worker (или обычного локального
    запуска) этот сервер не нужен и не запускается — см.
    ENABLE_HEALTH_SERVER.
    """

    if aiohttp_web is None:
        logger.error(
            "ENABLE_HEALTH_SERVER включён, но пакет aiohttp не "
            "установлен — добавь 'aiohttp' в requirements.txt"
        )
        return

    async def handle_ping(request):
        return aiohttp_web.Response(text="ok")

    app = aiohttp_web.Application()
    app.router.add_get("/", handle_ping)
    app.router.add_get("/health", handle_ping)

    runner = aiohttp_web.AppRunner(app)
    await runner.setup()

    site = aiohttp_web.TCPSite(runner, "0.0.0.0", HEALTH_SERVER_PORT)
    await site.start()

    logger.info(
        "Health-check сервер слушает порт %s (/ и /health)",
        HEALTH_SERVER_PORT,
    )


async def main():
    logger.info("=" * 60)
    logger.info("Запуск Userbot")
    logger.info("=" * 60)

    logger.info(
        "DeepSeek model=%s | thinking=%s",
        DEEPSEEK_MODEL,
        ENABLE_THINKING,
    )

    logger.info(
        "Delete tracker: save_messages=%s | save_media=%s | "
        "cache_max=%s",
        SAVE_DELETED_MESSAGES,
        SAVE_DELETED_MEDIA,
        DELETE_TRACKER_CACHE_MAX,
    )

    if ENABLE_HEALTH_SERVER:
        # Запускаем сразу, до client.start() — платформа должна
        # увидеть открытый порт как можно раньше при деплое.
        # Ошибка здесь (например, порт уже занят) не должна ронять
        # весь бот — это вспомогательная фича только для Render,
        # без неё юзербот всё равно должен нормально работать.
        try:
            await start_health_server()
        except Exception:
            logger.exception(
                "Не удалось запустить health-check сервер — "
                "продолжаю без него (порт %s уже занят?)",
                HEALTH_SERVER_PORT,
            )

    await client.start()

    me = await client.get_me()

    logger.info(
        "Авторизован: id=%s username=%s",
        me.id,
        me.username,
    )

    print()
    print("=" * 60)
    print("🐾 USERBOT ЗАПУЩЕН")
    print("=" * 60)
    print(
        f"Аккаунт: {me.first_name or ''} {me.last_name or ''}"
    )

    if me.username:
        print(f"Username: @{me.username}")

    print()
    print("Команды:")
    print(".catgirl")
    print(".tsundere")
    print(".clone (в группах — ответом на сообщение цели)")
    print(".reset")
    print(".mute 10m")
    print(".unmute")
    print(".remember")
    print(".ai запрос")
    print(".8ball вопрос")
    print(".decide A или B")
    print(".math 5 + 3")
    print(".stats")
    print(".auto here / .auto on / .auto off")
    print(".save (тумблер прямо в этом чате)")
    print(".f текст")
    print(".geo место / .geolive место")
    print(".ocr (ответом на фото)")
    print(".vm (ответом на голосовое/кружок)")
    print(".paid <звёзды> (ответом на фото)")
    print(".saveonce (ответом на одноразовое фото)")
    print(".info")
    print(".help")
    print()
    print(
        "Трекер удалённых сообщений: "
        + ("включён" if SAVE_DELETED_MESSAGES else "выключен")
        + " (SAVE_DELETED_MESSAGES в .env)"
    )
    print(
        "Сохранение медиа удалённых сообщений: "
        + ("включено" if SAVE_DELETED_MEDIA else "выключено")
        + " (SAVE_DELETED_MEDIA в .env)"
    )
    print(
        "Health-check сервер: "
        + (
            f"включён, порт {HEALTH_SERVER_PORT}"
            if ENABLE_HEALTH_SERVER
            else "выключен"
        )
    )
    print()
    print("Логи: logs/userbot.log")
    print("=" * 60)

    await client.run_until_disconnected()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        print("Userbot остановлен.")
    except Exception:
        logger.exception("КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ")
        raise
