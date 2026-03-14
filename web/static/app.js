let currentChatId = null;
let currentModelId = "plasma-1.0";
let isGenerating = false;

const sidebar = document.getElementById("sidebar");
const chatList = document.getElementById("chat-list");
const messagesEl = document.getElementById("messages");
const messagesInner = document.getElementById("messages-inner");
const input = document.getElementById("input");
const btnSend = document.getElementById("btn-send");
const btnNewChat = document.getElementById("btn-new-chat");
const sidebarToggle = document.getElementById("sidebar-toggle");
const modelSelect = document.getElementById("model-select");
const topbarTitle = document.getElementById("topbar-title");

async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

async function loadModels() {
  const models = await api("GET", "/api/models");
  modelSelect.innerHTML = "";
  models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name + (m.available ? "" : " — coming soon");
    opt.disabled = !m.available;
    modelSelect.appendChild(opt);
  });
  const available = models.find(m => m.available);
  if (available) {
    modelSelect.value = available.id;
    currentModelId = available.id;
  }
}

modelSelect.addEventListener("change", () => {
  currentModelId = modelSelect.value;
});

async function loadChats() {
  const chats = await api("GET", "/api/chats");
  chatList.innerHTML = "";
  chats.forEach(chat => {
    const el = document.createElement("div");
    el.className = "chat-item" + (chat.id === currentChatId ? " active" : "");
    el.innerHTML = `
      <div class="chat-item-info">
        <div class="chat-item-title">${escapeHtml(chat.title)}</div>
        <div class="chat-item-meta">${timeAgo(chat.updated_at)}</div>
      </div>
      <button class="chat-item-delete" aria-label="Delete chat">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    `;
    el.querySelector(".chat-item-info").addEventListener("click", () => selectChat(chat.id));
    el.querySelector(".chat-item-delete").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteChat(chat.id);
    });
    chatList.appendChild(el);
  });
}

async function selectChat(chatId) {
  currentChatId = chatId;
  loadChats();

  const msgs = await api("GET", `/api/chats/${chatId}/messages`);
  renderMessages(msgs);

  const chat = await api("GET", `/api/chats`);
  const c = chat.find(ch => ch.id === chatId);
  if (c) topbarTitle.textContent = c.title;

  input.focus();
}

async function createNewChat() {
  const { chat_id } = await api("POST", "/api/chats", { model_id: currentModelId });
  currentChatId = chat_id;
  await loadChats();
  renderMessages([]);
  topbarTitle.textContent = "1386.ai";
  input.focus();
  if (window.innerWidth <= 640) sidebar.classList.add("collapsed");
}

async function deleteChat(chatId) {
  await api("DELETE", `/api/chats/${chatId}`);
  if (currentChatId === chatId) {
    currentChatId = null;
    renderMessages([]);
    topbarTitle.textContent = "1386.ai";
  }
  loadChats();
}

function renderMessages(msgs) {
  messagesInner.innerHTML = "";
  if (msgs.length === 0) {
    messagesInner.appendChild(createEmptyState());
    return;
  }
  msgs.forEach(m => appendMessage(m.role, m.content));
  scrollToBottom();
}

function appendMessage(role, content) {
  const es = messagesInner.querySelector(".empty-state");
  if (es) es.remove();

  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.innerHTML = `
    <div class="message-role">${role === "user" ? "you" : "1386.ai"}</div>
    <div class="message-content">${escapeHtml(content)}</div>
  `;
  messagesInner.appendChild(el);
  scrollToBottom();
}

function showTyping() {
  const el = document.createElement("div");
  el.className = "message assistant";
  el.id = "typing-indicator";
  el.innerHTML = `
    <div class="message-role">1386.ai</div>
    <div class="typing">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  messagesInner.appendChild(el);
  scrollToBottom();
}

function hideTyping() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function createEmptyState() {
  const el = document.createElement("div");
  el.className = "empty-state";
  el.innerHTML = `
    <svg class="empty-state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
    </svg>
    <div class="empty-state-text">1386.ai</div>
    <div class="empty-state-sub">type a message below</div>
  `;
  return el;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text || isGenerating) return;

  if (!currentChatId) {
    const { chat_id } = await api("POST", "/api/chats", { model_id: currentModelId });
    currentChatId = chat_id;
    await loadChats();
  }

  isGenerating = true;
  input.value = "";
  autoResize();
  btnSend.disabled = true;

  appendMessage("user", text);
  showTyping();

  try {
    const { response } = await api("POST", `/api/chats/${currentChatId}/send`, {
      message: text,
      model_id: currentModelId,
    });
    hideTyping();
    appendMessage("assistant", response);
    loadChats();
  } catch (err) {
    hideTyping();
    appendMessage("assistant", `Error: ${err.message}`);
  } finally {
    isGenerating = false;
    updateSendButton();
    input.focus();
  }
}

input.addEventListener("input", () => {
  autoResize();
  updateSendButton();
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

btnSend.addEventListener("click", sendMessage);
btnNewChat.addEventListener("click", createNewChat);

function autoResize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 120) + "px";
}

function updateSendButton() {
  btnSend.disabled = !input.value.trim() || isGenerating;
}

sidebarToggle.addEventListener("click", () => {
  sidebar.classList.toggle("collapsed");
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function timeAgo(ts) {
  const diff = (Date.now() / 1000) - ts;
  if (diff < 60) return "now";
  if (diff < 3600) return Math.floor(diff / 60) + "m";
  if (diff < 86400) return Math.floor(diff / 3600) + "h";
  if (diff < 604800) return Math.floor(diff / 86400) + "d";
  return new Date(ts * 1000).toLocaleDateString();
}

async function init() {
  await loadModels();
  await loadChats();
  input.focus();
}

init();
