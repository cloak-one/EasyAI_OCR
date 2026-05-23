function byId(id) {
  return document.getElementById(id);
}

const els = {
  captureImage: byId("captureImage"),
  imageMeta: byId("imageMeta"),
  userText: byId("userText"),
  modelDelta: byId("modelDelta"),
  systemState: byId("systemState"),
  partialText: byId("partialText"),
  errorList: byId("errorList"),
  historyList: byId("historyList"),
};

let lastUpdatedAt = 0;
const PLACEHOLDER_SRC = "/placeholder.jpg";

if (els.captureImage) {
  els.captureImage.src = PLACEHOLDER_SRC;
}

function renderList(target, lines) {
  target.innerHTML = "";
  (lines || []).forEach((line) => {
    const li = document.createElement("li");
    li.textContent = typeof line === "string" ? line : JSON.stringify(line);
    target.appendChild(li);
  });
}

function renderHistory(items) {
  els.historyList.innerHTML = "";
  (items || []).forEach((item) => {
    const li = document.createElement("li");
    const user = (item && item.user) ? item.user : "";
    const model = (item && item.model) ? item.model : "";
    li.textContent = `U: ${user}\nA: ${model}`;
    els.historyList.appendChild(li);
  });
}

function renderState(state) {
  els.systemState.textContent = state.system_state || "IDLE";
  els.partialText.textContent = state.partial_text || "";
  els.userText.textContent = state.user_text || "";
  els.modelDelta.textContent = state.model_text_delta || "";
  renderList(els.errorList, state.errors || []);
  renderHistory(state.history || []);

  if (state.frame_ts) {
    const stamp = new Date(state.frame_ts * 1000).toLocaleTimeString();
    els.imageMeta.textContent = `最后图像时间: ${stamp}`;
    const frameUrl = `/api/frame.jpg?t=${state.frame_ts}`;
    if (els.captureImage.src !== frameUrl) {
      els.captureImage.src = frameUrl;
    }
  } else {
    els.imageMeta.textContent = "等待图像（占位图显示中）...";
    if (!els.captureImage.src || !els.captureImage.src.includes("/placeholder.jpg")) {
      els.captureImage.src = PLACEHOLDER_SRC;
    }
  }
}

async function pollState() {
  try {
    const res = await fetch("/api/state", { cache: "no-store" });
    if (!res.ok) {
      return;
    }
    const state = await res.json();
    const updatedAt = Number(state.updated_at || 0);
    if (updatedAt >= lastUpdatedAt) {
      lastUpdatedAt = updatedAt;
      renderState(state);
    }
  } catch (e) {
    // keep polling, no throw
  }
}

setInterval(pollState, 250);
pollState();
