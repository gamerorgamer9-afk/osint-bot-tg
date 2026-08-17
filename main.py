import asyncio
import os
import time
from collections import deque
from aiohttp import web, ClientSession
from telethon import TelegramClient, events
from google import genai
from google.genai import types

# ================= CONFIGURATION =================
API_ID = int(os.environ.get("API_ID", 0))
API_HASH = os.environ.get("API_HASH", "")
SESSION_STRING = os.environ.get("SESSION_STRING", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
PORT = int(os.environ.get("PORT", 8080))
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

# Инициализация клиентов
client = TelegramClient("userbot_session", API_ID, API_HASH)
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Состояние бота
bot_state = {
    "afk": False,
    "afk_reason": "",
    "afk_messages": [],
    "mode": None,  # "catgirl", "tsundere" или None
    "spam_active": False
}

# Кэш сообщений для мгновенного отслеживания удалений (до 500 сообщений)
msg_cache = deque(maxlen=500)

# ================= KEEP-ALIVE SERVER =================
async def handle_ping(request):
    return web.Response(text="OK - Userbot is active")

async def keep_alive():
    app = web.Application()
    app.router.add_get("/", handle_ping)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    
    # Автоматический пинг самого себя каждые 10 минут (чтобы Render не спал)
    if RENDER_EXTERNAL_URL:
        async with ClientSession() as session:
            while True:
                await asyncio.sleep(600)
                try:
                    await session.get(RENDER_EXTERNAL_URL)
                except Exception:
                    pass

# ================= GEMINI FAST GENERATION =================
async def fast_gemini_generate(prompt: str, system_instruction: str = None) -> str:
    if not ai_client:
        return "❌ GEMINI_API_KEY не настроен!"
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction or "Отвечай максимально кратко, чётко и по существу.",
        max_output_tokens=350,
        temperature=0.5
    )
    
    response = await ai_client.aio.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=config
    )
    return response.text

# ================= TELEGRAM HANDLERS =================

# 1. Кэширование всех входящих и исходящих сообщений для слежки за удалениями
@client.on(events.NewMessage)
async def cache_listener(event):
    if event.message and event.message.text:
        msg_cache.append(event.message)

# 2. Логгер удаленных сообщений
@client.on(events.MessageDeleted)
async def deleted_logger(event):
    for deleted_id in event.deleted_ids:
        for cached in list(msg_cache):
            if cached.id == deleted_id:
                sender = await cached.get_sender()
                sender_name = getattr(sender, 'first_name', 'Неизвестно')
                text = cached.text
                
                log_text = (
                    f"🗑 **Удалено сообщение!**\n"
                    f"👤 **От:** {sender_name} (`{cached.sender_id}`)\n"
                    f"💬 **Текст:** {text}"
                )
                await client.send_message("me", log_text)
                break

# 3. Базовые команды
@client.on(events.NewMessage(outgoing=True, pattern=r'^\.ping$'))
async def ping_handler(event):
    start = time.perf_counter()
    await event.edit("🏓 Pinging...")
    end = time.perf_counter()
    ms = round((end - start) * 1000, 2)
    await event.edit(f"🚀 **Pong!**\n⚡ Задержка: `{ms} ms`")

@client.on(events.NewMessage(outgoing=True, pattern=r'^\.help$'))
async def help_handler(event):
    help_text = (
        "⚙️ **Юзербот Команды:**\n\n"
        "🔹 `.ping` — Быстрая проверка отклика\n"
        "🔹 `.afk [причина]` — Включить AFK\n"
        "🔹 `.unafk` — Выключить AFK и показать сводку\n"
        "🔹 `.ai [текст]` — Быстрый ответ от Gemini (Stream)\n"
        "🔹 `.remember [кол-во]` — Быстрое саммари чата\n"
        "🔹 `.clone` — Скопировать стиль автора (в ответ на смс)\n"
        "🔹 `.catgirl` / `.tsundere` / `.reset` — Смена личности\n"
        "🔹 `.spam [кол-во] [текст]` — Спам с отменой\n"
        "🔹 `.stop` — Остановить спам"
    )
    await event.edit(help_text)

# 4. AFK Модуль
@client.on(events.NewMessage(outgoing=True, pattern=r'^\.afk(?:\s+(.+))?$'))
async def afk_on(event):
    reason = event.pattern_match.group(1) or "Занят"
    bot_state["afk"] = True
    bot_state["afk_reason"] = reason
    bot_state["afk_messages"] = []
    await event.edit(f"💤 **Режим AFK включен.**\nПричина: `{reason}`")

@client.on(events.NewMessage(outgoing=True, pattern=r'^\.unafk$'))
async def afk_off(event):
    if not bot_state["afk"]:
        return await event.edit("❌ Вы не в режиме AFK.")
    
    count = len(bot_state["afk_messages"])
    msg = f"🌅 **Режим AFK выключен.**\nПока вас не было, вам написали: `{count}` раз(а)."
    bot_state["afk"] = False
    bot_state["afk_reason"] = ""
    await event.edit(msg)

@client.on(events.NewMessage(incoming=True))
async def afk_listener(event):
    if bot_state["afk"] and (event.is_private or event.mentioned):
        sender = await event.get_sender()
        name = getattr(sender, 'first_name', 'Кто-то')
        bot_state["afk_messages"].append(f"{name}: {event.text}")
        await event.reply(f"🤖 Я сейчас AFK.\nПричина: `{bot_state['afk_reason']}`")

# 5. Gemini AI со СТРИМИНГОМ (Мгновенный вывод)
@client.on(events.NewMessage(outgoing=True, pattern=r'^\.ai\s+(.+)'))
async def ai_stream_handler(event):
    prompt = event.pattern_match.group(1)
    await event.edit("🧠 *Генерация...*")
    
    if not ai_client:
        return await event.edit("❌ Gemini API Key отсутствует!")
    
    try:
        response = await ai_client.aio.models.generate_content_stream(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=400, temperature=0.6)
        )
        
        full_text = ""
        last_update = time.time()
        
        async for chunk in response:
            full_text += chunk.text
            # Редактируем сообщение не чаще чем раз в 1.2 сек, чтобы не получить FloodWait
            if time.time() - last_update > 1.2:
                await event.edit(full_text + " ▌")
                last_update = time.time()
                
        await event.edit(full_text)
    except Exception as e:
        await event.edit(f"❌ Ошибка Gemini: `{str(e)}`")

# 6. Анализ истории (.remember)
@client.on(events.NewMessage(outgoing=True, pattern=r'^\.remember(?:\s+(\d+))?$'))
async def remember_handler(event):
    limit = int(event.pattern_match.group(1) or 25)
    await event.edit(f"📊 Анализирую последние {limit} сообщений...")
    
    raw_messages = await client.get_messages(event.chat_id, limit=limit)
    history = []
    for m in reversed(raw_messages):
        if m.text:
            name = m.sender.first_name if m.sender and hasattr(m.sender, 'first_name') else "User"
            history.append(f"{name}: {m.text}")
            
    context_text = "\n".join(history)
    prompt = f"Сделай краткое и понятное саммари этой переписки:\n\n{context_text}"
    
    res = await fast_gemini_generate(prompt)
    await event.edit(f"📝 **Краткая выжимка:**\n\n{res}")

# 7. Клонирование стиля (.clone)
@client.on(events.NewMessage(outgoing=True, pattern=r'^\.clone$'))
async def clone_handler(event):
    reply = await event.get_reply_message()
    if not reply or not reply.sender_id:
        return await event.edit("❌ Ответьте этой командой на сообщение пользователя!")
        
    await event.edit("🎭 Анализирую манеру речи...")
    user_msgs = await client.get_messages(event.chat_id, limit=40, from_user=reply.sender_id)
    texts = [m.text for m in user_msgs if m.text]
    
    if not texts:
        return await event.edit("❌ Недостаточно сообщений для анализа!")
        
    sample = "\n".join(texts[:15])
    prompt = f"Напиши одно короткое предложение в точности повторяя этот стиль, сленг и пунктуацию:\n\n{sample}"
    
    res = await fast_gemini_generate(prompt)
    await event.edit(res)

# 8. Режимы Личности
@client.on(events.NewMessage(outgoing=True, pattern=r'^\.(catgirl|tsundere|reset)$'))
async def mode_handler(event):
    cmd = event.pattern_match.group(1)
    if cmd == "reset":
        bot_state["mode"] = None
        await event.edit("🔄 Режим личности сброшен.")
    else:
        bot_state["mode"] = cmd
        await event.edit(f"🎭 Режим **{cmd}** успешно активирован!")

# 9. Спам с отменой
@client.on(events.NewMessage(outgoing=True, pattern=r'^\.spam\s+(\d+)\s+(.+)'))
async def spam_handler(event):
    count = int(event.pattern_match.group(1))
    text = event.pattern_match.group(2)
    await event.delete()
    
    bot_state["spam_active"] = True
    for _ in range(count):
        if not bot_state["spam_active"]:
            break
        await client.send_message(event.chat_id, text)
        await asyncio.sleep(0.3)
    bot_state["spam_active"] = False

@client.on(events.NewMessage(outgoing=True, pattern=r'^\.stop$'))
async def stop_spam(event):
    bot_state["spam_active"] = False
    await event.edit("⏹ Процесс остановлен.")

# ================= STARTUP =================
async def main():
    # Запуск внутреннего Keep-Alive веб-сервера
    asyncio.create_task(keep_alive())
    
    # Авторизация через SESSION_STRING без интерактивного ввода
    await client.start(phone=lambda: SESSION_STRING)
    print("✅ Юзербот успешно запущен и работает в разгоне!")
    await client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())