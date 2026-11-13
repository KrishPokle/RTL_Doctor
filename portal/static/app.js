"use strict";
// RTL Doctor portal: the chat on the left drives the dashboard on the right.
// The page keeps the conversation and the file; the server keeps nothing.

const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const state = {
  file: null,                   // {name, code}
  messages: [],                 // [{role, content}] as sent to the server
  limits: { file_types: [".v", ".sv"], max_file_bytes: 100000 },
  repair: null,                 // the last repair the server returned
  zoom: 1,
  busy: false,
};

const RAN = {                   // what each tool did, in the user's words
  lint_rtl: "Checked with the linter",
  render_schematic: "Drew the schematic",
  repair_rtl: "Asked the repair model",
};
const TAB_OF_TOOL = { lint_rtl: "findings", render_schematic: "schematic", repair_rtl: "repair" };
const MAX_HISTORY = 40;

// ------------------------------------------------------------------ chat
function addMessage(kind, text) {
  const node = el("div", "msg msg-" + kind, text);
  $("log").append(node);
  $("log").scrollTop = $("log").scrollHeight;
  return node;
}

async function ask(text) {
  text = text.trim();
  if (!text || state.busy) return;
  state.busy = true;
  $("send").disabled = true;
  $("starters").hidden = true;
  addMessage("user", text);
  state.messages.push({ role: "user", content: text });
  const reply = addMessage("assistant", "");
  reply.append(el("span", "working", "Working"));

  try {
    const response = await fetch("api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: state.messages.slice(-MAX_HISTORY), file: state.file }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(describeError(data.detail));
    state.messages.push({ role: "assistant", content: data.reply });
    reply.textContent = "";
    if (data.tools.length) {
      const ran = el("div", "ran");
      for (const tool of new Set(data.tools)) ran.append(el("span", "", RAN[tool] || tool));
      reply.append(ran);
    }
    reply.append(document.createTextNode(data.reply));
    if (data.findings) showFindings(data.findings);
    if (data.schematic) showSchematic(data.schematic);
    if (data.repair) showRepair(data.repair);
    const shown = data.tools.map((tool) => TAB_OF_TOOL[tool]).filter(Boolean).pop();
    if (shown) selectTab(shown);
  } catch (error) {
    state.messages.pop();                             // the question got no answer: do not resend it
    reply.className = "msg msg-error";
    reply.textContent = error.message || "The server did not answer.";
  }
  state.busy = false;
  $("send").disabled = false;
  $("log").scrollTop = $("log").scrollHeight;
}

function describeError(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join(" ");
  return "The server did not answer.";
}

// ------------------------------------------------------------------ the file
function chooseFile(file) {
  const name = file.name;
  if (!state.limits.file_types.some((type) => name.toLowerCase().endsWith(type))) {
    addMessage("error", `Only ${state.limits.file_types.join(" and ")} files are accepted.`);
    return;
  }
  if (file.size > state.limits.max_file_bytes) {
    addMessage("error", `${name} is ${file.size.toLocaleString()} bytes. The limit is ${state.limits.max_file_bytes.toLocaleString()}.`);
    return;
  }
  file.text().then((code) => useFile(name, code, `${name} is loaded.`));
}

function useFile(name, code, note) {
  state.file = { name, code };
  state.repair = null;
  const lines = code.replace(/\r\n?/g, "\n").split("\n");
  $("file-name").textContent = `${name}, ${lines.length} lines`;
  $("choose-file").textContent = "Change file";
  $("starters").hidden = state.messages.length > 0;
  addMessage("note", note);

  $("source").replaceChildren(...lines.map((line) => el("li", "", line)));
  $("findings-empty").hidden = true;
  $("findings-body").hidden = false;
  $("findings-count").hidden = true;
  $("summary-note").textContent = "Ask for a bug check to see the findings.";
  $("chart").replaceChildren();
  $("finding-list").replaceChildren();
  $("schematic-body").hidden = true;
  $("schematic-empty").hidden = false;
  $("repair-body").hidden = true;
  $("repair-empty").hidden = false;
  selectTab("findings");
}

// ------------------------------------------------------------------ dashboard
function selectTab(name) {
  for (const tab of document.querySelectorAll('[role="tab"]')) {
    const on = tab.id === "tab-" + name;
    tab.setAttribute("aria-selected", on);
    $(tab.getAttribute("aria-controls")).hidden = !on;
  }
  if (name === "schematic") fitSchematic();
}

function showFindings(findings) {
  const rows = $("source").children;
  for (const row of rows) row.className = "";
  for (const f of findings) {
    const row = rows[f.line - 1];
    if (row && row.className !== "error") row.className = f.severity;
  }
  $("findings-count").textContent = findings.length;
  $("findings-count").hidden = findings.length === 0;
  $("summary-note").textContent = findings.length
    ? `${findings.length} finding${findings.length === 1 ? "" : "s"}, by check`
    : "The linter found nothing wrong.";

  const counts = new Map();                           // check -> {n, error}
  for (const f of findings) {
    const key = f.code || "error";
    const entry = counts.get(key) || { n: 0, error: false };
    entry.n += 1;
    entry.error = entry.error || f.severity === "error";
    counts.set(key, entry);
  }
  const most = Math.max(1, ...[...counts.values()].map((entry) => entry.n));
  const chart = [];
  for (const [check, entry] of [...counts].sort((a, b) => b[1].n - a[1].n)) {
    const bar = el("span", "bar" + (entry.error ? " error" : ""));
    bar.style.width = `${(100 * entry.n) / most}%`;
    chart.push(el("span", "check", check), bar, el("span", "n", entry.n));
  }
  $("chart").replaceChildren(...chart);

  $("finding-list").replaceChildren(...findings.map((f) => {
    const button = el("button");
    button.type = "button";
    const place = f.line ? `Line ${f.line}` : "Whole file";
    button.append(el("span", "where " + f.severity, f.code ? `${place}, ${f.code}` : place), el("span", "text", f.message));
    button.addEventListener("click", () => {
      for (const row of rows) row.classList.remove("target");
      const row = rows[f.line - 1];
      if (row) {
        row.classList.add("target");
        row.scrollIntoView({ block: "center" });
      }
    });
    const item = el("li");
    item.append(button);
    return item;
  }));
}

function showSchematic(schematic) {
  const image = $("schematic-img");
  if (image.src) URL.revokeObjectURL(image.src);
  // an <img> cannot run scripts, whatever the SVG contains
  image.src = URL.createObjectURL(new Blob([schematic.svg], { type: "image/svg+xml" }));
  image.onload = fitSchematic;
  $("schematic-level").textContent = schematic.level === "gate"
    ? "Gate level: single gates and flip-flops"
    : "Block level: adders, multiplexers and registers";
  $("schematic-empty").hidden = true;
  $("schematic-body").hidden = false;
}

function zoomSchematic(zoom) {
  const image = $("schematic-img");
  state.zoom = Math.min(4, Math.max(0.1, zoom));
  image.style.width = `${image.naturalWidth * state.zoom}px`;
}

function fitSchematic() {
  const image = $("schematic-img");
  const room = $("viewer").clientWidth - 44;
  if (image.naturalWidth && room > 0) zoomSchematic(Math.min(1, room / image.naturalWidth));
}

function showRepair(repair) {
  state.repair = repair;
  $("repair-title").textContent = `Fix for a ${repair.bug_type.replaceAll("_", " ")} bug`;
  const lines = repair.diff.slice(2).map((line) => {  // the first two lines only name the two versions
    const kind = line.startsWith("@@") ? "hunk" : line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : "";
    return el("span", kind, line || " ");
  });
  $("diff").replaceChildren(...lines);
  $("use-repair").disabled = false;
  $("repair-empty").hidden = true;
  $("repair-body").hidden = false;
}

function repairedName() {
  return state.file.name.replace(/(_repaired)?(\.\w+)$/, "_repaired$2");
}

// ------------------------------------------------------------------ wiring
$("choose-file").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) => {
  if (event.target.files[0]) chooseFile(event.target.files[0]);
  event.target.value = "";                            // so the same file can be chosen again
});
$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("question").value;
  $("question").value = "";
  ask(text);
});
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});
for (const button of $("starters").children) button.addEventListener("click", () => ask(button.textContent));
for (const tab of document.querySelectorAll('[role="tab"]')) {
  tab.addEventListener("click", () => selectTab(tab.id.slice(4)));
}
$("zoom-in").addEventListener("click", () => zoomSchematic(state.zoom * 1.25));
$("zoom-out").addEventListener("click", () => zoomSchematic(state.zoom / 1.25));
$("zoom-fit").addEventListener("click", fitSchematic);
$("use-repair").addEventListener("click", () => {
  const repair = state.repair;
  useFile(repairedName(), repair.patched, "Now working on the repaired file.");
});
$("download-repair").addEventListener("click", () => {
  const link = el("a");
  link.href = URL.createObjectURL(new Blob([state.repair.patched], { type: "text/plain" }));
  link.download = repairedName();
  link.click();
  URL.revokeObjectURL(link.href);
});

fetch("api/status").then((response) => response.json()).then((status) => {
  state.limits = status;
  if (!status.model) {
    $("model-note").textContent = "No language model is connected. A stand-in answers: it runs the tools, but it cannot explain what they find.";
    $("model-note").hidden = false;
  }
});
