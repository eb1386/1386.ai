/* 1386.ai — chat client.
   Every icon is a dot matrix declared below, not an icon font. Each dot
   carries an index (--i) so CSS can stagger, sweep or chase through it.
   The server returns a whole completion, so the reveal is paced here. */

const $ = (id) => document.getElementById(id);

const main = $("main");
const side = $("side");
const veil = $("veil");
const scroll = $("scroll");
const thread = $("thread");
const heroAsk = $("hero-ask");
const tipsEl = $("tips");
const chatsEl = $("chats");
const noChats = $("no-chats");
const findEl = $("find");
const box = $("box");
const input = $("input");
const sendBtn = $("send");
const note = $("note");
const pickBtn = $("pick-btn");
const pickMenu = $("pick-menu");
const pickName = $("pick-name");
const tempBtn = $("temp-toggle");
const tempFlag = $("tempflag");

const softMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let chatId = null;
let modelId = null;
let models = [];
let chats = [];
let busy = false;
let temporary = false;
let tempTurns = [];
let query = "";

const TIP_COUNT = 3;

/* ── the matrix ──────────────────────────────────────────────
   '#' lights a cell, 'o' lights it in the accent, '.' stays dark. */

/* the wordmark is the logo, so it needs a real 7-row pixel font.
   glyphs carry their own width and are joined with a one-column gap. */
const FONT = {
  "1": [".#.",
        "##.",
        ".#.",
        ".#.",
        ".#.",
        ".#.",
        "###"],
  "3": ["#####",
        "....#",
        "....#",
        ".####",
        "....#",
        "....#",
        "#####"],
  "8": [".###.",
        "#...#",
        "#...#",
        ".###.",
        "#...#",
        "#...#",
        ".###."],
  "6": [".###.",
        "#....",
        "#....",
        "####.",
        "#...#",
        "#...#",
        ".###."],
  ".": ["..",
        "..",
        "..",
        "..",
        "..",
        "##",
        "##"],
  a:   [".....",
        ".....",
        ".###.",
        "....#",
        ".####",
        "#...#",
        ".####"],
  i:   [".#.",
        "...",
        "##.",
        ".#.",
        ".#.",
        ".#.",
        "###"],
};

// everything from accentFrom onward lights in the accent colour
function pixelText(str, accentFrom) {
  const rows = Array.from({ length: 7 }, () => []);
  [...str].forEach((ch, i) => {
    const glyph = FONT[ch];
    if (!glyph) return;
    const hot = accentFrom !== undefined && i >= accentFrom;
    if (i > 0) rows.forEach((r) => r.push("."));
    glyph.forEach((line, y) => {
      [...line].forEach((c) => rows[y].push(c === "#" ? (hot ? "o" : "#") : "."));
    });
  });
  return rows.map((r) => r.join(""));
}

const ICONS = {
  panel: ["#######",
          "#.#...#",
          "#.#...#",
          "#.#...#",
          "#.#...#",
          "#.#...#",
          "#######"],
  plus:  ["...#...",
          "...#...",
          "...#...",
          "#######",
          "...#...",
          "...#...",
          "...#..."],
  find:  [".####..",
          "#....#.",
          "#....#.",
          "#....#.",
          ".####..",
          ".....#.",
          "......#"],
  // a ring that never closes: nothing here is kept
  temp:  ["..###..",
          ".#...#.",
          "#.....#",
          "#......",
          "#.....#",
          ".#...#.",
          "..###.."],
  pin:   ["..###..",
          "..#.#..",
          "..###..",
          "...#...",
          "...#...",
          "...#...",
          "...#..."],
  x:     ["#.....#",
          ".#...#.",
          "..#.#..",
          "...#...",
          "..#.#..",
          ".#...#.",
          "#.....#"],
  send:  ["...#...",
          "..###..",
          ".#.#.#.",
          "#..#..#",
          "...#...",
          "...#...",
          "...#..."],
  chev:  [".......",
          ".......",
          "#.....#",
          ".#...#.",
          "..#.#..",
          "...#...",
          "......."],
  tick:  [".......",
          "......#",
          ".....#.",
          "#...#..",
          ".#.#...",
          "..#....",
          "......."],
  arrow: [".......",
          "...#...",
          "....#..",
          "#######",
          "....#..",
          "...#...",
          "......."],
  copy:  ["..#####",
          "..#...#",
          "..#...#",
          "###...#",
          "#.#####",
          "#.....#",
          "#######"],
  eb:    ["#####.####.",
          "#.....#...#",
          "#.....#...#",
          "####..####.",
          "#.....#...#",
          "#.....#...#",
          "#####.####."],
};

function dotMatrix(rows, opts) {
  if (!rows || !rows.length) return "";
  const pitch = 3.2, dot = 2.4, pad = 0.4;

  const cells = [];
  rows.forEach((row, y) => {
    [...row].forEach((ch, x) => {
      if (ch !== ".") cells.push({ x, y, core: ch === "o" });
    });
  });

  // order the indices around the centre so a chase reads as circular
  if (opts && opts.radial) {
    const cx = (rows[0].length - 1) / 2;
    const cy = (rows.length - 1) / 2;
    cells.sort((a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx));
  }

  const cells_svg = cells.map((c, i) =>
    `<rect${c.core ? ' class="hot"' : ""} style="--i:${i}" ` +
    `x="${(pad + c.x * pitch).toFixed(2)}" y="${(pad + c.y * pitch).toFixed(2)}" ` +
    `width="${dot}" height="${dot}"/>`).join("");

  const w = (pad * 2 + (rows[0].length - 1) * pitch + dot).toFixed(2);
  const h = (pad * 2 + (rows.length - 1) * pitch + dot).toFixed(2);
  return `<svg class="dots" viewBox="0 0 ${w} ${h}" aria-hidden="true">${cells_svg}</svg>`;
}

function dotIcon(name, opts) {
  return dotMatrix(ICONS[name], opts);
}

// the logo, wherever it appears: 1386 in ink, .ai in accent
const wordmark = () => dotMatrix(pixelText("1386.ai", 4));
const LOGO_SM = '<span class="logo logo--sm">' + wordmark() + "</span>";

function paintIcons(root) {
  (root || document).querySelectorAll("[data-icon]").forEach((el) => {
    el.innerHTML = dotIcon(el.dataset.icon, { radial: el.dataset.order === "radial" });
  });
}

// same font, stacked two-up so it survives a 16px tab
function setFavicon() {
  const top = pixelText("13"), bot = pixelText("86");
  const width = Math.max(top[0].length, bot[0].length);
  const pad = (rows) => rows.map((r) => {
    const gap = width - r.length;
    return ".".repeat(Math.floor(gap / 2)) + r + ".".repeat(Math.ceil(gap / 2));
  });
  const rows = [...pad(top), ".".repeat(width), ...pad(bot)];

  let cells = "";
  rows.forEach((row, y) => {
    [...row].forEach((ch, x) => {
      if (ch !== ".") cells += `<rect x="${x * 1.2}" y="${y * 1.2}" width="1" height="1"/>`;
    });
  });
  const w = width * 1.2, h = rows.length * 1.2;
  const side = Math.max(w, h);
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${(w - side) / 2} ${(h - side) / 2} ${side} ${side}">` +
    `<g fill="#e05200">${cells}</g></svg>`;
  const link = document.querySelector('link[rel="icon"]');
  if (link) link.href = "data:image/svg+xml," + encodeURIComponent(svg);
}

/* ── api ─────────────────────────────────────────────────── */

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

/* ── render helpers ──────────────────────────────────────── */

function esc(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

function markdown(text) {
  try {
    if (typeof marked === "undefined") return esc(text);
    const block = (tex) => {
      try { return '<div class="math-block">' + katex.renderToString(tex.trim(), { displayMode: true }) + "</div>"; }
      catch { return '<div class="math-block"><code>' + esc(tex) + "</code></div>"; }
    };
    const inline = (tex) => {
      try { return katex.renderToString(tex.trim(), { displayMode: false }); }
      catch { return "<code>" + esc(tex) + "</code>"; }
    };
    let p = text;
    p = p.replace(/\$\$([\s\S]*?)\$\$/g, (_, t) => block(t));
    p = p.replace(/\\\[([\s\S]*?)\\\]/g, (_, t) => block(t));
    p = p.replace(/\$([^\$\n]+?)\$/g, (_, t) => inline(t));
    p = p.replace(/\\\(([\s\S]*?)\\\)/g, (_, t) => inline(t));
    return marked.parse(p);
  } catch {
    return esc(text);
  }
}

function addMsg(role, content) {
  const el = document.createElement("div");
  el.className = "msg msg--" + (role === "user" ? "user" : "bot");
  if (role === "user") {
    el.innerHTML = '<div class="txt"></div>';
    el.querySelector(".txt").textContent = content;
  } else {
    el.innerHTML = LOGO_SM + '<div class="txt"></div>';
    el.querySelector(".txt").innerHTML = markdown(content);
    if (content) el.appendChild(copyBar(content));
  }
  thread.appendChild(el);
  toBottom();
  return el;
}

function copyBar(text) {
  const bar = document.createElement("div");
  bar.className = "tools";
  bar.innerHTML =
    '<button class="copy" type="button">' + dotIcon("copy") + "<span>Copy</span></button>";
  const btn = bar.querySelector(".copy");
  btn.addEventListener("click", async () => {
    const label = btn.querySelector("span");
    try {
      await navigator.clipboard.writeText(text);
      label.textContent = "Copied";
    } catch {
      label.textContent = "Ctrl+C";
    }
    setTimeout(() => { label.textContent = "Copy"; }, 1600);
  });
  return bar;
}

function toBottom() {
  scroll.scrollTo({ top: scroll.scrollHeight, behavior: softMotion ? "auto" : "smooth" });
}

function setEmpty(isEmpty) {
  main.classList.toggle("is-empty", isEmpty);
  $(isEmpty ? "slot-hero" : "slot-dock").appendChild(box);
  note.textContent = isEmpty ? "" : "Small model · can be wrong";
  if (isEmpty) drawTips();
}

function showThread(msgs) {
  thread.innerHTML = "";
  if (!msgs.length) { setEmpty(true); return; }
  setEmpty(false);
  msgs.forEach((m) => addMsg(m.role, m.content));
}

/* ── thinking + reveal ───────────────────────────────────── */

function showThinking() {
  const el = document.createElement("div");
  el.className = "msg msg--bot";
  el.id = "thinking";
  // 3x3, delayed by row+col so the pulse crosses it diagonally
  let dots = "";
  for (let i = 0; i < 9; i++) {
    dots += `<i style="--i:${Math.floor(i / 3) + (i % 3)}"></i>`;
  }
  el.innerHTML =
    LOGO_SM +
    '<div class="txt"><div class="think">' +
    '<span class="grid" aria-hidden="true">' + dots + "</span>" +
    '<span class="think__word">Thinking</span>' +
    '<span class="think__time">0.0s</span></div></div>';
  thread.appendChild(el);

  const started = Date.now();
  const time = el.querySelector(".think__time");
  el._timer = setInterval(() => {
    const s = (Date.now() - started) / 1000;
    time.textContent = s < 60
      ? s.toFixed(1) + "s"
      : Math.floor(s / 60) + "m " + String(Math.floor(s % 60)).padStart(2, "0") + "s";
  }, 100);

  toBottom();
}

function stopThinking() {
  const el = $("thinking");
  if (!el) return;
  clearInterval(el._timer);
  el.remove();
}

function reveal(el, text) {
  return new Promise((resolve) => {
    const txt = el.querySelector(".txt");
    const done = () => {
      txt.innerHTML = markdown(text);
      el.appendChild(copyBar(text));
      toBottom();
      resolve();
    };
    if (softMotion) return done();

    const parts = text.match(/\S+\s*/g) || [text];
    let i = 0, sofar = "";
    const near = () => scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 140;

    (function tick() {
      if (i >= parts.length) return done();
      sofar += parts[i++];
      txt.innerHTML = esc(sofar) + '<span class="caret" aria-hidden="true"></span>';
      if (near()) scroll.scrollTop = scroll.scrollHeight;
      setTimeout(tick, 15 + Math.random() * 20);
    })();
  });
}

/* ── model picker ────────────────────────────────────────── */

function drawPicker() {
  pickMenu.innerHTML = "";
  models.forEach((m) => {
    const o = document.createElement("button");
    o.type = "button";
    o.className = "opt";
    o.setAttribute("role", "option");
    o.setAttribute("aria-selected", String(m.id === modelId));
    if (!m.available) o.setAttribute("aria-disabled", "true");
    o.innerHTML =
      '<span class="opt__body"><span class="opt__name"></span>' +
      '<span class="opt__sub"></span></span>' +
      '<span class="opt__tick">' + dotIcon("tick") + "</span>";
    o.querySelector(".opt__name").textContent = m.name;
    o.querySelector(".opt__sub").textContent = m.available ? m.params : "not trained yet";
    if (m.available) o.addEventListener("click", () => { choose(m.id); closePick(); });
    pickMenu.appendChild(o);
  });
}

function choose(id) {
  const m = models.find((x) => x.id === id);
  if (!m) return;
  modelId = id;
  pickName.textContent = m.name.split(/\s+[—-]\s+/)[0];
  drawPicker();
}

// drops from under the button, nudged only as far as the viewport demands
function placePick() {
  const b = pickBtn.getBoundingClientRect();
  const m = pickMenu.getBoundingClientRect();
  const gap = 6, edge = 8;
  const want = b.bottom + gap;
  const top = Math.max(edge, Math.min(want, window.innerHeight - m.height - edge));
  const left = Math.max(edge, Math.min(b.right - m.width, window.innerWidth - m.width - edge));
  pickMenu.style.top = top + "px";
  pickMenu.style.left = left + "px";
  // docked at the very bottom there is no room below, so it rises instead
  pickMenu.classList.toggle("up", top < want - 1);
}

function openPick() {
  drawPicker();
  pickMenu.hidden = false;
  pickBtn.setAttribute("aria-expanded", "true");
  placePick();
  (pickMenu.querySelector('[aria-selected="true"]') || pickMenu.firstElementChild)
    ?.classList.add("cursor");
}
function closePick() {
  pickMenu.hidden = true;
  pickBtn.setAttribute("aria-expanded", "false");
  pickMenu.querySelectorAll(".cursor").forEach((n) => n.classList.remove("cursor"));
}
function movePick(step) {
  const items = [...pickMenu.querySelectorAll('.opt:not([aria-disabled="true"])')];
  if (!items.length) return;
  const at = items.findIndex((n) => n.classList.contains("cursor"));
  items.forEach((n) => n.classList.remove("cursor"));
  const next = items[(at + step + items.length) % items.length] || items[0];
  next.classList.add("cursor");
  next.scrollIntoView({ block: "nearest" });
}

pickBtn.addEventListener("click", () => { pickMenu.hidden ? openPick() : closePick(); });
pickBtn.addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (pickMenu.hidden) openPick();
    else movePick(e.key === "ArrowDown" ? 1 : -1);
  } else if (e.key === "Enter" && !pickMenu.hidden) {
    e.preventDefault();
    pickMenu.querySelector(".cursor")?.click();
  } else if (e.key === "Escape") {
    closePick();
  }
});
document.addEventListener("click", (e) => {
  if (!pickMenu.hidden && !$("pick").contains(e.target)) closePick();
});
// a fixed menu would drift away from its button otherwise
scroll.addEventListener("scroll", () => { if (!pickMenu.hidden) closePick(); });

/* ── chat list: pinned first, then by age ────────────────── */

function bucket(ts) {
  const days = (Date.now() / 1000 - ts) / 86400;
  if (days < 1) return "Today";
  if (days < 2) return "Yesterday";
  if (days < 7) return "Last 7 days";
  if (days < 30) return "Last 30 days";
  return "Older";
}

function chatRow(c) {
  const el = document.createElement("div");
  el.className = "chat" + (c.id === chatId ? " on" : "");
  el.innerHTML =
    '<button class="chat__go" type="button"><span class="chat__name"></span></button>' +
    `<button class="chat__act chat__act--pin" type="button" aria-pressed="${!!c.pinned}"
       aria-label="${c.pinned ? "Unpin chat" : "Pin chat"}">${dotIcon("pin")}</button>` +
    `<button class="chat__act chat__act--x" type="button" aria-label="Delete chat">${dotIcon("x")}</button>`;
  el.querySelector(".chat__name").textContent = c.title;
  el.querySelector(".chat__go").addEventListener("click", () => openChat(c.id));
  el.querySelector(".chat__act--pin").addEventListener("click", async (e) => {
    e.stopPropagation();
    await api("PATCH", `/api/chats/${c.id}/pin`, { pinned: !c.pinned });
    loadChats();
  });
  el.querySelector(".chat__act--x").addEventListener("click", (e) => {
    e.stopPropagation();
    removeChat(c.id);
  });
  return el;
}

function drawChats() {
  const q = query.trim().toLowerCase();
  const shown = q ? chats.filter((c) => c.title.toLowerCase().includes(q)) : chats;

  chatsEl.innerHTML = "";
  if (!shown.length) {
    noChats.hidden = false;
    noChats.textContent = q ? `Nothing matches "${query.trim()}".` : "No chats yet.";
    return;
  }
  noChats.hidden = true;

  let group = null;
  shown.forEach((c) => {
    const name = c.pinned ? "Pinned" : bucket(c.updated_at);
    if (name !== group) {
      group = name;
      const h = document.createElement("div");
      h.className = "group";
      h.innerHTML = '<span class="tag"></span>';
      h.querySelector(".tag").textContent = name;
      chatsEl.appendChild(h);
    }
    chatsEl.appendChild(chatRow(c));
  });
}

async function loadModels() {
  models = await api("GET", "/api/models");
  const order = ["plasma-1.1", "plasma-1.0"];
  const first =
    order.map((id) => models.find((m) => m.id === id && m.available)).find(Boolean) ||
    models.find((m) => m.available) ||
    models[0];
  if (first) choose(first.id);
}

async function loadChats() {
  chats = await api("GET", "/api/chats");
  drawChats();
}

async function openChat(id) {
  setTemporary(false);
  chatId = id;
  drawChats();
  showThread(await api("GET", `/api/chats/${id}/messages`));
  if (isNarrow()) openSide(false);
  input.focus();
}

async function newChat(temp) {
  setTemporary(!!temp);
  chatId = null;
  tempTurns = [];
  thread.innerHTML = "";
  setEmpty(true);
  drawChats();
  if (isNarrow()) openSide(false);
  input.focus();
}

async function removeChat(id) {
  await api("DELETE", `/api/chats/${id}`);
  if (chatId === id) {
    chatId = null;
    thread.innerHTML = "";
    setEmpty(true);
  }
  loadChats();
}

function setTemporary(on) {
  temporary = on;
  tempBtn.setAttribute("aria-pressed", String(on));
  tempFlag.hidden = !on;
  heroAsk.textContent = on ? "Temporary chat" : "Where should we begin?";
  input.placeholder = on ? "This chat will not be saved" : "Ask 1386";
}

/* ── send ────────────────────────────────────────────────── */

async function send(text) {
  if (!text || busy) return;

  busy = true;
  input.value = "";
  grow();
  refreshSend();

  if (main.classList.contains("is-empty")) setEmpty(false);
  addMsg("user", text);
  showThinking();

  try {
    let reply;
    if (temporary) {
      const out = await api("POST", "/api/generate", {
        message: text, model_id: modelId, history: tempTurns,
      });
      reply = out.response;
      tempTurns.push({ role: "user", content: text });
      tempTurns.push({ role: "assistant", content: reply });
    } else {
      if (!chatId) {
        const made = await api("POST", "/api/chats", { model_id: modelId });
        chatId = made.chat_id;
      }
      const out = await api("POST", `/api/chats/${chatId}/send`, {
        message: text, model_id: modelId,
      });
      reply = out.response;
    }
    stopThinking();
    await reveal(addMsg("assistant", ""), reply);
    if (!temporary) loadChats();
  } catch (err) {
    stopThinking();
    const el = addMsg("assistant", "");
    el.querySelector(".txt").innerHTML =
      "<p>" + esc(err.message) + "</p>" +
      '<p style="color:var(--fg-3);font-size:0.86em">The model runs locally. ' +
      "Check that the server is up, then send again.</p>";
  } finally {
    busy = false;
    refreshSend();
    input.focus();
  }
}

/* ── tips ────────────────────────────────────────────────── */

// fresh three every time the empty state comes back
function drawTips() {
  tipsEl.innerHTML = "";
  const picks = window.PROMPTS
    ? window.PROMPTS.pick(TIP_COUNT)
    : ["What is gravity?", "Explain photosynthesis in simple terms.", "Why does it rain?"];
  picks.forEach((t, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tip";
    b.style.animationDelay = 320 + i * 60 + "ms";
    b.innerHTML = '<span class="tip__txt"></span><span class="tip__go">' + dotIcon("arrow") + "</span>";
    b.querySelector(".tip__txt").textContent = t;
    b.addEventListener("click", () => send(t));
    tipsEl.appendChild(b);
  });
}

/* ── composer ────────────────────────────────────────────── */

function grow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}
function refreshSend() { sendBtn.disabled = !input.value.trim() || busy; }

input.addEventListener("input", () => { grow(); refreshSend(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input.value.trim()); }
});
box.addEventListener("submit", (e) => { e.preventDefault(); send(input.value.trim()); });

findEl.addEventListener("input", () => { query = findEl.value; drawChats(); });
findEl.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { findEl.value = ""; query = ""; drawChats(); }
});

/* ── sidebar ─────────────────────────────────────────────── */

const isNarrow = () => window.innerWidth <= 820;

function openSide(open) {
  side.classList.toggle("shut", !open);
  document.body.classList.toggle("side-open", open);
}

$("side-close").addEventListener("click", () => openSide(false));
$("side-open").addEventListener("click", () => openSide(true));
veil.addEventListener("click", () => openSide(false));
$("new-chat").addEventListener("click", () => newChat(false));
tempBtn.addEventListener("click", () => {
  if (temporary) { setTemporary(false); return; }
  newChat(true);
});

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!pickMenu.hidden) { closePick(); pickBtn.focus(); }
  else if (isNarrow() && document.body.classList.contains("side-open")) openSide(false);
});

// crossing the breakpoint must not strand an overlay sidebar on the thread
let narrow = isNarrow();
function syncWidth() {
  document.body.classList.toggle("narrow", isNarrow());
  if (isNarrow() !== narrow) { narrow = isNarrow(); openSide(!narrow); }
}
window.addEventListener("resize", () => {
  syncWidth();
  if (!pickMenu.hidden) closePick();
});

/* ── boot ────────────────────────────────────────────────── */

(async function start() {
  paintIcons();
  setFavicon();
  $("brand").innerHTML = '<span class="logo logo--nav">' + wordmark() + "</span>";
  $("hero-mark").innerHTML = '<span class="logo logo--hero">' + wordmark() + "</span>";
  $("pfp").innerHTML = dotIcon("eb");

  syncWidth();
  openSide(!isNarrow());
  setEmpty(true);
  setTemporary(false);
  refreshSend();
  try {
    await loadModels();
    await loadChats();
  } catch {
    note.textContent = "Cannot reach the server";
  }
  input.focus();
})();
