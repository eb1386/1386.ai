# chat history storage

import sqlite3
import time
import uuid
from pathlib import Path

DB_PATH = Path(__file__).parent / "chat_history.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New chat',
            model_id TEXT NOT NULL DEFAULT 'plasma-1.0',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
    """)
    conn.commit()
    conn.close()


def create_chat(model_id="plasma-1.0"):
    chat_id = uuid.uuid4().hex[:12]
    now = time.time()
    conn = get_db()
    conn.execute(
        "INSERT INTO chats (id, title, model_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (chat_id, "New chat", model_id, now, now),
    )
    conn.commit()
    conn.close()
    return chat_id


def list_chats():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, model_id, created_at, updated_at FROM chats ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_chat(chat_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def rename_chat(chat_id, title):
    conn = get_db()
    conn.execute("UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                 (title, time.time(), chat_id))
    conn.commit()
    conn.close()


def delete_chat(chat_id):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()


def add_message(chat_id, role, content):
    now = time.time()
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, now),
    )
    # auto-title from first message
    if role == "user":
        chat = conn.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if chat and chat["title"] == "New chat":
            title = content[:50].strip()
            if len(content) > 50:
                title += "..."
            conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))
    conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
    conn.commit()
    conn.close()


def get_messages(chat_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
