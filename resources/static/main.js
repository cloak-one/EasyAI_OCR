const proto = location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${proto}://${location.host}/ws_ui`);

const convList = document.getElementById("conv-list");
const captureList = document.getElementById("capture-list");
const messageCount = document.getElementById("message-count");
const captureCount = document.getElementById("capture-count");
const conversationScroller = convList.parentElement;

let partialEl = null;

initEmptyStates();
refreshMessageCount();
refreshCaptureCount();

ws.onmessage = ({ data }) => {
  if (data.startsWith("INIT:")) {
    const init = JSON.parse(data.slice(5));
    init.captures?.forEach((b64) => appendCapture(b64));
    init.finals?.forEach((text) => appendClassifiedMessage(text));
    if (init.device_state) {
      updateState(init.device_state);
    }
  } else if (data.startsWith("PARTIAL:")) {
    const text = data.slice(8);
    const classification = classifyMessage(text);
    if (!partialEl) {
      partialEl = appendMsg(classification.role, classification.text, true);
    } else {
      partialEl.className = `msg ${classification.role} partial`;
      updateMsgText(partialEl, classification.text);
    }
  } else if (data.startsWith("FINAL:")) {
    if (partialEl) {
      partialEl.remove();
      partialEl = null;
      refreshMessageCount();
      ensureEmptyState(convList, "conversation");
    }
    appendClassifiedMessage(data.slice(6));
  } else if (data.startsWith("CAPTURE:")) {
    appendCapture(data.slice(8));
  } else if (data.startsWith("STATE:")) {
    updateState(JSON.parse(data.slice(6)));
  }
};

ws.onclose = () => {
  setStatusValue("s-camera", "离线");
  setStatusValue("s-audio", "离线");
  setStatusValue("s-asr", "连接中断");
  setStatusValue("s-intent", "idle");
  setCardState("card-camera", "offline");
  setCardState("card-audio", "offline");
  setCardState("card-asr", "warning");
  setCardState("card-intent", "neutral");
};

function initEmptyStates() {
  ensureEmptyState(convList, "conversation");
  ensureEmptyState(captureList, "capture");
}

function ensureEmptyState(container, type) {
  const hasContent = container.querySelector(type === "conversation" ? ".msg" : ".capture-item");
  const empty = container.querySelector(".empty-state");

  if (hasContent && empty) {
    empty.remove();
    return;
  }

  if (!hasContent && !empty) {
    const state = document.createElement("div");
    state.className = "empty-state";
    if (type === "conversation") {
      state.innerHTML = "<div><strong>等待对话数据</strong><span>语音识别和系统响应会实时出现在这里。</span></div>";
    } else {
      state.innerHTML = "<div><strong>等待截图数据</strong><span>命令触发后的关键画面会按时间显示在这里。</span></div>";
    }
    container.appendChild(state);
  }
}

function appendClassifiedMessage(rawText) {
  const classification = classifyMessage(rawText);
  return appendMsg(classification.role, classification.text, false);
}

function appendMsg(role, text, isPartial = false) {
  clearEmptyState(convList);
  const el = document.createElement("article");
  el.className = `msg ${role}${isPartial ? " partial" : ""}`;
  el.innerHTML = `
    <div class="msg-body"></div>
    <div class="msg-meta">
      <time>${getTimeLabel()}</time>
    </div>
  `;
  updateMsgText(el, text);
  convList.appendChild(el);
  conversationScroller.scrollTop = conversationScroller.scrollHeight;
  refreshMessageCount();
  return el;
}

function updateMsgText(el, text) {
  const body = el.querySelector(".msg-body");
  if (body) {
    body.textContent = text;
  }
}

function appendCapture(b64) {
  clearEmptyState(captureList);
  const item = document.createElement("article");
  item.className = "capture-item";
  item.innerHTML = `
    <img src="data:image/jpeg;base64,${b64}" alt="命令截图">
    <div class="capture-meta">
      <span class="capture-label">命令截图</span>
      <time class="capture-time">${getTimeLabel()}</time>
    </div>
  `;
  captureList.prepend(item);
  refreshCaptureCount();
}

function clearEmptyState(container) {
  const empty = container.querySelector(".empty-state");
  if (empty) {
    empty.remove();
  }
}

function updateState(s) {
  setStatusValue("s-camera", s.camera_connected ? "在线" : "离线");
  setStatusValue("s-audio", s.audio_connected ? "在线" : "离线");
  setStatusValue("s-asr", s.asr_active ? "识别中" : "待机");
  setStatusValue("s-intent", s.current_intent || "idle");

  setCardState("card-camera", s.camera_connected ? "online" : "offline");
  setCardState("card-audio", s.audio_connected ? "online" : "offline");
  setCardState("card-asr", s.asr_active ? "active" : "neutral");
  setCardState("card-intent", s.current_intent ? "active" : "neutral");
}

function setStatusValue(id, value) {
  const target = document.getElementById(id);
  if (target) {
    target.textContent = value;
  }
}

function setCardState(id, state) {
  const card = document.getElementById(id);
  if (card) {
    card.dataset.state = state;
  }
}

function refreshMessageCount() {
  const count = convList.querySelectorAll(".msg").length;
  messageCount.textContent = `${count} 条`;
}

function refreshCaptureCount() {
  const count = captureList.querySelectorAll(".capture-item").length;
  captureCount.textContent = `${count} 条`;
}

function classifyMessage(rawText) {
  const text = String(rawText || "").trim();
  if (text.startsWith("[AI]")) {
    return {
      role: "ai",
      text: text.replace(/^\[AI\]\s*/, ""),
    };
  }
  if (text.startsWith("[页码]")) {
    return {
      role: "ai",
      text: text.replace(/^\[页码\]\s*/, ""),
    };
  }
  return { role: "user", text };
}

function getTimeLabel() {
  return new Date().toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
