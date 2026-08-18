# chat api server

import asyncio
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from web.db import (init_db, create_chat, list_chats, get_chat, rename_chat,
                    delete_chat, add_message, get_messages, set_pinned)
from web.model_manager import ModelManager

app = FastAPI(title="1386.ai Chat")
manager = ModelManager()

STATIC_DIR = Path(__file__).parent / "static"


class FreshStatic(StaticFiles):
    """runs locally, so never let a stale css or js linger in the cache"""

    def is_not_modified(self, response_headers, request_headers):
        return False

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


app.mount("/static", FreshStatic(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/models")
def api_models():
    return manager.get_available_models()


@app.get("/api/runtime")
def api_runtime():
    """what this instance is actually running on"""
    return {"device": manager.device.type, "host": "self-hosted"}


@app.get("/api/chats")
def api_list_chats():
    return list_chats()


class CreateChatBody(BaseModel):
    # model_ prefix is reserved by pydantic
    model_config = {"protected_namespaces": ()}
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


class PinChatBody(BaseModel):
    pinned: bool


@app.patch("/api/chats/{chat_id}/pin")
def api_pin_chat(chat_id: str, body: PinChatBody):
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    set_pinned(chat_id, body.pinned)
    return {"ok": True, "pinned": body.pinned}


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
    model_config = {"protected_namespaces": ()}
    message: str
    model_id: str = "plasma-1.0"


@app.post("/api/chats/{chat_id}/send")
async def api_send_message(chat_id: str, body: SendMessageBody):
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")

    if not body.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    # prior turns only, the builder appends this message itself
    history = get_messages(chat_id)
    add_message(chat_id, "user", body.message.strip())

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


class GenerateBody(BaseModel):
    model_config = {"protected_namespaces": ()}
    message: str
    model_id: str = "plasma-1.0"
    history: List[Dict[str, str]] = []


@app.post("/api/generate")
async def api_generate(body: GenerateBody):
    """temporary chat, nothing is written to the database"""
    if not body.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    try:
        response = await asyncio.to_thread(
            manager.generate, body.model_id, body.message.strip(),
            history=body.history,
        )
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model not available: {e}")
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    return {"response": response}
