# chat api server

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from web.db import init_db, create_chat, list_chats, get_chat, rename_chat, delete_chat, add_message, get_messages
from web.model_manager import ModelManager

app = FastAPI(title="1386.ai Chat")
manager = ModelManager()

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/models")
def api_models():
    return manager.get_available_models()


@app.get("/api/chats")
def api_list_chats():
    return list_chats()


class CreateChatBody(BaseModel):
    model_id: str = "plasma-1.0"


@app.post("/api/chats")
def api_create_chat(body: CreateChatBody):
    chat_id = create_chat(body.model_id)
    return {"chat_id": chat_id}


class RenameChatBody(BaseModel):
    title: str


@app.patch("/api/chats/{chat_id}")
def api_rename_chat(chat_id: str, body: RenameChatBody):
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    rename_chat(chat_id, body.title)
    return {"ok": True}


@app.delete("/api/chats/{chat_id}")
def api_delete_chat(chat_id: str):
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    delete_chat(chat_id)
    return {"ok": True}


@app.get("/api/chats/{chat_id}/messages")
def api_get_messages(chat_id: str):
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    return get_messages(chat_id)


class SendMessageBody(BaseModel):
    message: str
    model_id: str = "plasma-1.0"


@app.post("/api/chats/{chat_id}/send")
async def api_send_message(chat_id: str, body: SendMessageBody):
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")

    if not body.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    add_message(chat_id, "user", body.message.strip())
    history = get_messages(chat_id)

    try:
        response = await asyncio.to_thread(
            manager.generate, body.model_id, body.message.strip(),
            history=history,
        )
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model not available: {e}")
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    add_message(chat_id, "assistant", response)

    return {"response": response}
