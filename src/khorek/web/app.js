/* khorek trace viewer — vanilla JS, no build step. */

const state = {
  runs: [],
  run: null,
  events: [],
  bySeq: new Map(),
  open: new Set(),
  expandAll: false,
  live: true,
  filter: { q: "", agent: "", kind: "" },
  focused: -1,
  pollTimer: null,
  loadingRun: false,
};

const $ = (selector, root = document) => root.querySelector(selector);

/* ---------- formatting ---------- */

function esc(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmtTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString(undefined, {
    month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function fmtClock(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function fmtDuration(ms) {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s % 60)}s`;
}

function fmtBytes(bytes) {
  if (bytes === null || bytes === undefined) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MiB`;
}

function shortArgs(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  if (!text) return "";
  return text.length > 110 ? text.slice(0, 110) + "…" : text;
}

/* ---------- syntax highlighted JSON ---------- */

const JSON_TOKEN = /("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"\s*:)|("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*")|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;

function highlightJson(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  let out = "";
  let last = 0;
  for (const match of text.matchAll(JSON_TOKEN)) {
    out += esc(text.slice(last, match.index));
    const token = match[0];
    let cls = "j-num";
    if (match[1]) cls = "j-key";
    else if (match[2]) cls = "j-string";
    else if (match[3]) cls = "j-lit";
    out += `<span class="${cls}">${esc(token)}</span>`;
    last = match.index + token.length;
  }
  out += esc(text.slice(last));
  return out;
}

function codeBlock(value, { tall = false, plain = false } = {}) {
  const pre = document.createElement("pre");
  pre.className = "code" + (tall ? " tall" : "") + (plain ? " plain" : "");
  if (plain) pre.textContent = String(value ?? "");
  else pre.innerHTML = highlightJson(value);
  return pre;
}

/* ---------- tiny DOM helpers ---------- */

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function kvList(rows) {
  const grid = el("div", { class: "kv" });
  for (const [key, value] of rows) {
    if (value === null || value === undefined || value === "") continue;
    grid.append(el("div", { class: "k", text: key }), el("div", { class: "v", text: String(value) }));
  }
  return grid;
}

function section(title, body, { open = false, flush = false, actions = [] } = {}) {
  const details = el("details", { class: "section" + (flush ? " flush" : "") });
  if (open) details.open = true;
  const summary = el("summary", { text: title });
  if (actions.length) {
    const bar = el("span", { class: "actions" });
    for (const action of actions) bar.append(action);
    summary.append(bar);
  }
  details.append(summary);
  const content = el("div", { class: "section-body" });
  if (body) content.append(body);
  details.append(content);
  return details;
}

function chip(text, cls = "") {
  return el("span", { class: "chip " + cls, text });
}

function button(label, onclick, cls = "ghost") {
  return el("button", { class: cls, text: label, onclick });
}

/* ---------- data access ---------- */

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body.error) detail = body.error;
    } catch { /* ignore */ }
    throw new Error(`${response.status} ${detail}`);
  }
  return response.json();
}

function toast(text) {
  const node = $("#toast");
  node.textContent = text;
  node.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.remove("show"), 1400);
}

async function copy(text, label = "copied") {
  try {
    await navigator.clipboard.writeText(text);
    toast(label);
  } catch {
    toast("clipboard unavailable");
  }
}

/* ---------- runs sidebar ---------- */

function isRunning(run) {
  if (run.meta.ended_at) return false;
  return Date.now() / 1000 - run.updated_at < 15 * 60;
}

function renderRuns() {
  const list = $("#run-list");
  list.replaceChildren();
  $("#run-count").textContent = String(state.runs.length);
  for (const run of state.runs) {
    const meta = run.meta || {};
    const item = el("button", {
      class: "run-item" + (state.run && state.run.id === run.id ? " selected" : ""),
      onclick: () => selectRun(run.id),
    });
    const line1 = el("div", { class: "line1" });
    if (isRunning(run)) line1.append(el("span", { class: "running", title: "run in progress" }));
    line1.append(el("span", { class: "command", text: meta.command || "run" }));
    line1.append(el("span", { class: "when", text: fmtClock(meta.started_at) }));
    item.append(line1);
    item.append(el("div", { class: "model", text: meta.model || "—" }));
    const counts = el("div", { class: "counts" });
    counts.append(el("span", { text: `${run.events} events` }));
    if (run.errors) counts.append(el("span", { class: "badge-error", text: `${run.errors} errors` }));
    else counts.append(el("span", { text: "no errors" }));
    item.append(counts);
    list.append(item);
  }
}

/* ---------- run header ---------- */

function statusOf(meta, run) {
  if (isRunning(run)) return ["running", "running"];
  if (meta.status === "fatal") return ["fatal", "fatal"];
  if (meta.status === "issues" || meta.exit_code === 1) return ["issues", "exit 1"];
  if (meta.exit_code === 0) return ["ok", "exit 0"];
  if (meta.exit_code !== null && meta.exit_code !== undefined) return ["error", `exit ${meta.exit_code}`];
  return ["", "incomplete"];
}

function renderHeader() {
  const header = $("#run-header");
  header.replaceChildren();
  if (!state.run) return;
  const meta = state.run.meta || {};
  const [statusClass, statusText] = statusOf(meta, state.run);

  const title = el("div", { class: "run-title" });
  title.append(el("h1", { text: meta.command || "run" }));
  title.append(el("span", { class: "status-pill " + statusClass, text: statusText }));
  title.append(el("span", { class: "run-id", text: state.run.id }));
  title.append(button("copy id", () => copy(state.run.id, "run id copied")));
  const download = el("a", {
    class: "button-link",
    href: `/api/runs/${encodeURIComponent(state.run.id)}/raw`,
    download: `${state.run.id}.events.jsonl`,
    text: "download .jsonl",
  });
  title.append(download);
  header.append(title);

  const grid = el("div", { class: "meta-grid" });
  const cells = [
    ["started", fmtTime(meta.started_at)],
    ["ended", fmtTime(meta.ended_at)],
    ["duration", fmtDuration(meta.duration_ms)],
    ["model", meta.model || "—"],
    ["exit code", meta.exit_code ?? "—"],
    ["pid", meta.pid ?? "—"],
    ["events", state.run.events],
    ["errors", state.run.errors],
    ["project", meta.project_root || "—"],
    ["database", meta.db_path || "—"],
  ];
  for (const [key, value] of cells) {
    grid.append(
      el("div", { class: "meta-cell" },
        el("div", { class: "k", text: key }),
        el("div", { class: "v", text: String(value) }),
      ),
    );
  }
  header.append(grid);

  if (Array.isArray(meta.argv) && meta.argv.length) {
    header.append(section("argv", codeBlock(meta.argv.join(" "), { plain: true })));
  }
}

/* ---------- event rendering ---------- */

function eventError(event) {
  if (event.error) {
    if (typeof event.error === "string") return event.error;
    return [event.error.code, event.error.message].filter(Boolean).join(": ");
  }
  if (event.kind === "agent_end" && event.status === "error") {
    return [event.error?.code, event.error?.message].filter(Boolean).join(": ") || "agent failed";
  }
  return null;
}

function eventOpen(event) {
  return state.expandAll || state.open.has(event.seq);
}

function renderMessages(messages, { diffBase = null } = {}) {
  const container = el("div");
  if (!Array.isArray(messages)) return codeBlock(messages);
  const previous = diffBase ? (diffBase.request?.messages || []) : null;
  messages.forEach((message, index) => {
    const isNew = previous
      ? index >= previous.length || JSON.stringify(previous[index]) !== JSON.stringify(message)
      : false;
    const node = el("div", { class: "msg" + (isNew ? " diff-new" : "") });
    const head = el("div", { class: "msg-head" });
    const role = message?.role || "unknown";
    head.append(el("span", { class: "role " + role, text: role }));
    if (message?.tool_call_id) head.append(el("span", { class: "msg-meta", text: message.tool_call_id }));
    if (message?.name) head.append(el("span", { class: "msg-meta", text: message.name }));
    if (isNew) head.append(chip("new", "good"));
    if (typeof message?.content === "string") {
      head.append(el("span", { class: "msg-meta", text: `${message.content.length} chars` }));
    }
    node.append(head);
    const body = el("div", { class: "msg-body" });
    let content = message?.content;
    if (Array.isArray(content)) {
      content = content
        .map((part) => (part && typeof part === "object" && "text" in part ? part.text : JSON.stringify(part)))
        .join("\n");
    }
    if (typeof content === "string" && content.length) {
      body.append(codeBlock(content, { plain: true }));
    }
    if (Array.isArray(message?.tool_calls)) {
      for (const call of message.tool_calls) {
        const card = el("div", { class: "toolcall" });
        card.append(el("div", { class: "name", text: call?.function?.name || call?.name || "tool" }));
        const args = call?.function?.arguments ?? call?.arguments;
        let parsed = args;
        if (typeof args === "string") {
          try { parsed = JSON.parse(args); } catch { parsed = args; }
        }
        card.append(el("div", { class: "args" }, codeBlock(parsed)));
        body.append(card);
      }
    }
    node.append(body);
    container.append(node);
  });
  return container;
}

function renderTools(tools) {
  const container = el("div");
  if (!Array.isArray(tools) || !tools.length) return el("div", { class: "muted", text: "no tools" });
  for (const tool of tools) {
    const fn = tool.function || tool;
    const details = el("details", { class: "tool-def" });
    const summary = el("summary");
    summary.append(el("span", { class: "name", text: fn.name }));
    if (fn.description) summary.append(el("span", { class: "desc", text: fn.description }));
    details.append(summary);
    const body = el("div", { class: "tool-body" });
    body.append(codeBlock(fn.parameters ?? {}));
    details.append(body);
    container.append(details);
  }
  return container;
}

function renderToolCall(call, result, error) {
  const container = el("div");
  const name = call?.name || "tool";
  const id = call?.id;
  container.append(kvList([["tool", name], ["call id", id]]));
  let parsed = call?.arguments;
  if (typeof parsed === "string") {
    try { parsed = JSON.parse(parsed); } catch { parsed = call.arguments; }
  }
  container.append(section("arguments", codeBlock(parsed), { open: true }));
  if (error) {
    container.append(el("div", { class: "error-banner", text: error }));
  } else if (result !== undefined) {
    container.append(section("result", codeBlock(result), { open: true }));
  }
  return container;
}

function toolCallSummary(event) {
  const call = event.call || {};
  return `${call.name || "tool"} ${shortArgs(call.arguments)}`;
}

function eventSummary(event) {
  if (event.kind === "model_call") {
    const messages = event.request?.messages?.length ?? 0;
    const tools = event.request?.tools?.length ?? 0;
    const calls = (event.tool_calls || []).map((call) => call.name).filter(Boolean);
    const bits = [`${messages} msgs`, `${tools} tools`];
    if (calls.length) bits.push(`calls ${calls.join(", ")}`);
    else if (event.content) bits.push("text reply");
    else if (event.thinking) bits.push("thinking only");
    return bits.join(" · ");
  }
  if (event.kind === "tool_call") return toolCallSummary(event);
  if (event.kind === "agent_start") return `${event.tools?.length ?? 0} tools · ${event.max_tool_calls} max calls`;
  if (event.kind === "agent_end") return event.status + (event.error?.code ? ` · ${event.error.code}` : "");
  if (event.kind === "repair") return `${event.tool}: ${event.error}`;
  if (event.kind === "text_reply") return `text #${event.consecutive}`;
  if (event.kind === "run_start") return event.command || "";
  if (event.kind === "run_end") return event.status || "";
  const keys = Object.keys(event).filter((key) => !["seq", "ts", "kind", "agent", "step"].includes(key));
  return keys.slice(0, 3).map((key) => `${key}=${shortArgs(event[key])}`).join(" · ");
}

function eventBody(event, previousSameAgent) {
  const body = el("div");
  const error = eventError(event);

  if (event.kind === "model_call") {
    const request = event.request || {};
    const finish = event.finish_reason;
    if (error) body.append(el("div", { class: "error-banner", text: error }));
    if (finish === "length") {
      body.append(el("div", { class: "warn-banner", text:
        "finish_reason=length: the model hit max_tokens before finishing. A thinking model that never calls a tool will look exactly like this." }));
    }
    const overview = el("div", { class: "actions" });
    body.append(
      kvList([
        ["agent", event.agent],
        ["request step", event.step],
        ["duration", fmtDuration(event.duration_ms)],
        ["http status", event.status],
        ["finish reason", finish],
        ["tool_choice", request.tool_choice],
        ["temperature", request.temperature],
        ["max_tokens", request.max_tokens],
        ["reasoning_effort", request.reasoning_effort],
        ["request bytes", fmtBytes(event.request_bytes)],
        ["prompt tokens", event.usage?.prompt_tokens],
        ["completion tokens", event.usage?.completion_tokens],
        ["total tokens", event.usage?.total_tokens],
      ]),
    );

    if (event.thinking) {
      const think = el("div", { class: "thinking", text: event.thinking });
      body.append(section(`thinking (${event.thinking.length.toLocaleString()} chars)`, think, { open: true }));
    }
    if (event.content) {
      body.append(section("content", el("div", { class: "content-block", text: event.content }), { open: true }));
    }
    if (Array.isArray(event.tool_calls) && event.tool_calls.length) {
      const calls = el("div");
      for (const call of event.tool_calls) {
        const card = el("div", { class: "toolcall" });
        card.append(el("div", { class: "name", text: call.name || "tool" }));
        let parsed = call.arguments;
        if (typeof parsed === "string") {
          try { parsed = JSON.parse(parsed); } catch { parsed = call.arguments; }
        }
        card.append(codeBlock(parsed));
        calls.append(card);
      }
      body.append(section(`tool calls (${event.tool_calls.length})`, calls, { open: true }));
    }

    const messages = request.messages || [];
    if (previousSameAgent) {
      const previousMessages = previousSameAgent.request?.messages || [];
      const diff = [];
      messages.forEach((message, index) => {
        if (index >= previousMessages.length || JSON.stringify(previousMessages[index]) !== JSON.stringify(message)) {
          diff.push(message);
        }
      });
      const note = el("div", { class: "diff-note", text:
        `${diff.length} message(s) new or changed since the previous call for ${event.agent} (#${previousSameAgent.seq}).` });
      const diffNode = el("div");
      diffNode.append(note);
      diffNode.append(renderMessages(diff));
      body.append(section("new since previous call", diffNode));
    }
    body.append(
      section(
        `messages (${messages.length})`,
        renderMessages(messages, { diffBase: previousSameAgent }),
        { open: !previousSameAgent },
      ),
    );
    body.append(section(`tools (${(request.tools || []).length})`, renderTools(request.tools)));

    const raw = el("div");
    raw.append(codeBlock(request));
    body.append(section("raw request", raw));

    const responseValue = event.response ?? { message: event.message };
    const rawResponse = el("div");
    rawResponse.append(codeBlock(responseValue, { tall: true }));
    body.append(section("raw response", rawResponse));

    const actions = el("div", { class: "actions" });
    actions.append(button("copy request", () => copy(JSON.stringify(request, null, 2), "request copied")));
    actions.append(button("copy response", () => copy(JSON.stringify(responseValue, null, 2), "response copied")));
    actions.append(button("copy event", () => copy(JSON.stringify(event, null, 2), "event copied")));
    body.append(actions);
    return body;
  }

  if (event.kind === "tool_call") {
    if (error) body.append(el("div", { class: "error-banner", text: error }));
    body.append(renderToolCall(event.call, event.result, error));
    body.append(kvList([["duration", fmtDuration(event.duration_ms)], ["agent", event.agent], ["step", event.step]]));
    return body;
  }

  if (event.kind === "agent_start") {
    body.append(
      kvList([
        ["agent", event.agent],
        ["max tool calls", event.max_tool_calls],
        ["force tool call", event.force_tool_call],
        ["text is final", event.text_is_final],
        ["terminal tool", event.terminal_tool],
        ["system prompt bytes", fmtBytes(event.system_prompt_bytes)],
        ["user prompt bytes", fmtBytes(event.user_prompt_bytes)],
      ]),
    );
    body.append(section("system prompt", codeBlock(event.system_prompt || "", { plain: true, open: true })));
    body.append(section("user prompt", codeBlock(event.user_prompt || "", { plain: true })));
    body.append(section(`tools (${(event.tools || []).length})`, renderTools(event.tools)));
    return body;
  }

  if (event.kind === "agent_end") {
    if (error) body.append(el("div", { class: "error-banner", text: error }));
    body.append(
      kvList([
        ["agent", event.agent],
        ["status", event.status],
        ["duration", fmtDuration(event.duration_ms)],
        ["error code", event.error?.code],
        ["error message", event.error?.message],
        ["last text", event.error?.last_text],
      ]),
    );
    if (event.text) body.append(section("final text", codeBlock(event.text, { plain: true }), { open: true }));
    if (event.data !== null && event.data !== undefined) body.append(section("final data", codeBlock(event.data), { open: true }));
    return body;
  }

  if (event.kind === "run_start" || event.kind === "run_end") {
    const copy = { ...event };
    delete copy.seq;
    delete copy.ts;
    delete copy.kind;
    body.append(codeBlock(copy));
    return body;
  }

  if (event.kind === "repair" || event.kind === "text_reply") {
    const copy = { ...event };
    delete copy.seq;
    delete copy.ts;
    body.append(codeBlock(copy));
    return body;
  }

  const copy = { ...event };
  delete copy.seq;
  delete copy.ts;
  body.append(codeBlock(copy));
  return body;
}

function renderEventCard(event, index, previousSameAgent) {
  const error = eventError(event);
  const card = el("article", {
    class: "event" + (error ? " error" : "") + (eventOpen(event) ? " open" : ""),
    "data-seq": String(event.seq),
    "data-index": String(index),
  });

  const head = el("button", { class: "event-head", onclick: () => toggleEvent(event.seq, card) });
  head.append(el("span", { class: "caret", text: "▶" }));
  head.append(el("span", { class: "seq", text: `#${event.seq}` }));
  head.append(el("span", { class: "kind " + event.kind, text: event.kind }));
  if (event.agent) head.append(el("span", { class: "agent", text: event.agent }));
  head.append(el("span", { class: "summary", text: eventSummary(event) }));

  const right = el("span", { class: "right" });
  if (event.duration_ms !== undefined && event.duration_ms !== null) {
    right.append(chip(fmtDuration(event.duration_ms)));
  }
  if (event.kind === "model_call") {
    if (event.thinking) right.append(chip(`think ${event.thinking.length}`));
    if (event.tool_calls?.length) right.append(chip(`${event.tool_calls.length} call${event.tool_calls.length > 1 ? "s" : ""}`, "good"));
    if (event.finish_reason) right.append(chip(event.finish_reason, event.finish_reason === "length" ? "warn" : ""));
    if (event.error) right.append(chip(event.error.code || "error", "bad"));
  }
  if (event.kind === "agent_end") {
    right.append(chip(event.status, event.status === "ok" ? "good" : "bad"));
  }
  if (error) right.append(chip("error", "bad"));
  head.append(right);
  card.append(head);

  const body = el("div", { class: "event-body" });
  card.append(body);
  // Render lazily on first open; traces can be very large.
  card._body = body;
  card._event = event;
  card._previous = previousSameAgent;
  if (card.classList.contains("open")) {
    body.append(eventBody(event, previousSameAgent));
    card._rendered = true;
  }
  return card;
}

function toggleEvent(seq, card) {
  if (state.open.has(seq)) state.open.delete(seq);
  else state.open.add(seq);
  card.classList.toggle("open");
  if (!card._rendered) {
    card._body.append(eventBody(card._event, card._previous));
    card._rendered = true;
  }
}

function visibleEvents() {
  const query = state.filter.q.trim().toLowerCase();
  return state.events.filter((event) => {
    if (state.filter.agent && event.agent !== state.filter.agent) return false;
    if (state.filter.kind && event.kind !== state.filter.kind) return false;
    if (query && !JSON.stringify(event).toLowerCase().includes(query)) return false;
    return true;
  });
}

function renderTimeline() {
  const timeline = $("#timeline");
  timeline.replaceChildren();
  const events = visibleEvents();
  $("#filter-count").textContent = `${events.length} / ${state.events.length} events`;
  if (!events.length) {
    timeline.append(el("div", { class: "empty", text: "No events match the current filter." }));
    return;
  }
  const previousByAgent = new Map();
  for (const event of events) {
    const previous = event.agent ? previousByAgent.get(event.agent) : null;
    if (event.kind === "model_call" && event.agent) previousByAgent.set(event.agent, event);
    timeline.append(renderEventCard(event, timeline.children.length, previous));
  }
}

function updateFilterOptions() {
  const agents = [...new Set(state.events.map((event) => event.agent).filter(Boolean))].sort();
  const kinds = [...new Set(state.events.map((event) => event.kind).filter(Boolean))].sort();
  const agentSelect = $("#filter-agent");
  const kindSelect = $("#filter-kind");
  const keepAgent = state.filter.agent;
  const keepKind = state.filter.kind;
  agentSelect.replaceChildren(el("option", { value: "", text: "all agents" }));
  kindSelect.replaceChildren(el("option", { value: "", text: "all kinds" }));
  for (const agent of agents) agentSelect.append(el("option", { value: agent, text: agent }));
  for (const kind of kinds) kindSelect.append(el("option", { value: kind, text: kind }));
  agentSelect.value = keepAgent;
  kindSelect.value = keepKind;
}

/* ---------- run selection / polling ---------- */

function updateLive() {
  const live = $("#live");
  live.classList.toggle("on", state.live);
  live.classList.toggle("off", !state.live);
  $("#live-label").textContent = state.live ? "live" : "paused";
}

async function loadRuns() {
  try {
    const data = await fetchJson("/api/runs");
    $("#trace-root").textContent = data.root;
    state.runs = data.runs;
    renderRuns();
    if (state.run) {
      const fresh = state.runs.find((run) => run.id === state.run.id);
      if (fresh) {
        state.run = fresh;
        renderHeader();
      }
    }
  } catch (error) {
    $("#live-label").textContent = "disconnected";
  }
}

async function selectRun(id) {
  state.loadingRun = true;
  try {
    state.run = await fetchJson(`/api/runs/${encodeURIComponent(id)}`);
    state.events = [];
    state.bySeq.clear();
    state.open.clear();
    state.expandAll = false;
    state.focused = -1;
    const data = await fetchJson(`/api/runs/${encodeURIComponent(id)}/events`);
    for (const event of data.events) {
      state.events.push(event);
      state.bySeq.set(event.seq, event);
      if (event.kind === "model_call" && !state.open.size) state.open.add(event.seq);
    }
    $("#empty").classList.add("hidden");
    $("#run-view").classList.remove("hidden");
    history.replaceState(null, "", `#${encodeURIComponent(id)}`);
    renderHeader();
    updateFilterOptions();
    renderTimeline();
    renderRuns();
  } catch (error) {
    toast(String(error));
  } finally {
    state.loadingRun = false;
  }
}

async function refreshSelected({ reset = false } = {}) {
  if (!state.run || state.loadingRun) return;
  try {
    const id = encodeURIComponent(state.run.id);
    if (reset) {
      state.events = [];
      state.bySeq.clear();
    }
    const since = state.events.length ? state.events[state.events.length - 1].seq : 0;
    const data = await fetchJson(`/api/runs/${id}/events?since=${since}`);
    if (data.events.length) {
      for (const event of data.events) {
        if (state.bySeq.has(event.seq)) continue;
        state.events.push(event);
        state.bySeq.set(event.seq, event);
      }
      updateFilterOptions();
      renderTimeline();
      renderHeader();
    }
  } catch (error) {
    toast(String(error));
  }
}

async function tick() {
  if (!state.live) return;
  await loadRuns();
  if (state.run) await refreshSelected();
}

/* ---------- keyboard ---------- */

function moveFocus(delta) {
  const cards = [...document.querySelectorAll(".event")];
  if (!cards.length) return;
  state.focused = Math.max(0, Math.min(cards.length - 1, state.focused + delta));
  cards.forEach((card, index) => card.classList.toggle("focused", index === state.focused));
  cards[state.focused].scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function setupKeyboard() {
  document.addEventListener("keydown", (event) => {
    const active = document.activeElement;
    const typing = active && (active.tagName === "INPUT" || active.tagName === "SELECT" || active.tagName === "TEXTAREA");
    if (event.key === "/" && !typing) {
      event.preventDefault();
      $("#search").focus();
      return;
    }
    if (event.key === "Escape") {
      if (typing) { active.blur(); return; }
      state.open.clear();
      state.expandAll = false;
      renderTimeline();
      return;
    }
    if (typing) return;
    if (event.key === "j") { event.preventDefault(); moveFocus(1); }
    if (event.key === "k") { event.preventDefault(); moveFocus(-1); }
    if (event.key === "Enter") {
      const card = document.querySelectorAll(".event")[state.focused];
      if (card) {
        const seq = Number(card.dataset.seq);
        toggleEvent(seq, card);
      }
    }
    if (event.key === "e") {
      state.expandAll = !state.expandAll;
      $("#expand-all").textContent = state.expandAll ? "Collapse all" : "Expand all";
      renderTimeline();
    }
  });
}

/* ---------- wiring ---------- */

function wire() {
  $("#search").addEventListener("input", (event) => {
    state.filter.q = event.target.value;
    renderTimeline();
  });
  $("#filter-agent").addEventListener("change", (event) => {
    state.filter.agent = event.target.value;
    renderTimeline();
  });
  $("#filter-kind").addEventListener("change", (event) => {
    state.filter.kind = event.target.value;
    renderTimeline();
  });
  $("#expand-all").addEventListener("click", () => {
    state.expandAll = !state.expandAll;
    $("#expand-all").textContent = state.expandAll ? "Collapse all" : "Expand all";
    renderTimeline();
  });
  $("#refresh").addEventListener("click", async () => {
    await loadRuns();
    await refreshSelected();
  });
  $("#live").addEventListener("click", () => {
    state.live = !state.live;
    updateLive();
    if (state.live) tick();
  });
  $("#live").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") $("#live").click();
  });
  setupKeyboard();
}

async function boot() {
  wire();
  updateLive();
  await loadRuns();
  const hash = decodeURIComponent(location.hash.slice(1));
  const target = state.runs.find((run) => run.id === hash) || state.runs[0];
  if (target) await selectRun(target.id);
  state.pollTimer = setInterval(tick, 2000);
}

boot();
