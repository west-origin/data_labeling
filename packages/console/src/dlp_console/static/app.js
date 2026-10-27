// 운영 콘솔 화면 (ADR 0061). 빌드 단계 없는 순수 JS. 모든 글은 textContent로 넣는다 (HTML 주입 없음).
"use strict";

// ── 토큰: 콘솔 주소의 #token=… 을 읽어 세션 저장소에 두고 주소창에서 지운다 ──
const TOKEN = (() => {
  const m = location.hash.match(/token=([A-Za-z0-9_\-]+)/);
  let t = m ? m[1] : null;
  try {
    if (t) sessionStorage.setItem("dlp-console-token", t);
    else t = sessionStorage.getItem("dlp-console-token");
  } catch (e) { /* 저장소를 못 쓰면 주소의 토큰만 쓴다 */ }
  if (m) history.replaceState(null, "", location.pathname);
  return t || "";
})();

const state = { overview: null, current: null, pollTimer: null, theme: null };

// ── 작은 도우미 ──
// replaceChildren에 null을 넘기면 "null" 글자가 들어가므로 거른다
function fill(el, ...children) { el.replaceChildren(...children.flat().filter((c) => c !== null && c !== undefined && c !== false)); }

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

async function api(path, opts = {}) {
  const headers = { "X-Console-Token": TOKEN, ...(opts.headers || {}) };
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, { ...opts, headers });
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  if (!res.ok) {
    const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : res.statusText;
    throw new Error(detail);
  }
  return data;
}

function banner(text, isError) {
  const b = document.getElementById("banner");
  b.textContent = text;
  b.className = "banner" + (isError ? " error" : "");
  b.hidden = !text;
}

const STATUS_KO = {
  done: "완료", ready: "실행 가능", blocked: "대기", skipped: "건너뜀",
  running: "실행 중", succeeded: "성공", failed: "실패", interrupted: "중단됨",
  open: "열림", collected: "수집함", revoked: "회수함",
  candidate: "후보", passed: "게이트 통과(승인 대기)", deployed: "배포됨", rejected: "게이트 실패", retired: "은퇴",
};
const pill = (s) => h("span", { class: "pill " + s }, STATUS_KO[s] || s);

// ── 테마 (config/review_ui.yaml) ──
function applyTheme(theme) {
  state.theme = theme;
  let mode = theme.default_mode;
  try { mode = localStorage.getItem("dlp-console-mode") || mode; } catch (e) { /* 기본 모드 */ }
  setMode(mode);
  document.getElementById("brand-name").textContent = "운영 콘솔";
  document.getElementById("logo").alt = theme.name;
  document.getElementById("logo").src = theme.logo;
}

function setMode(mode) {
  const colors = state.theme.modes[mode] || state.theme.modes.dark;
  for (const [k, v] of Object.entries(colors)) document.documentElement.style.setProperty("--" + k, v);
  document.documentElement.dataset.mode = mode;
  document.getElementById("mode").textContent = mode === "dark" ? "화이트" : "다크";
  try { localStorage.setItem("dlp-console-mode", mode); } catch (e) { /* 기억하지 못해도 된다 */ }
}

// ── 탭 ──
function showTab(name) {
  for (const tab of document.querySelectorAll('[role="tab"]')) {
    const on = tab.id === "tab-" + name;
    tab.setAttribute("aria-selected", on ? "true" : "false");
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !on;
  }
  if (name === "training") loadTraining();
  if (name === "pretrain") loadPretrain();
  if (name === "jobs") loadJobs();
}

// ── 세션 목록 ──
async function loadOverview() {
  const data = await api("/api/overview");
  state.overview = data;
  if (!state.theme) applyTheme(data.theme);
  const list = document.getElementById("session-list");
  list.replaceChildren();
  for (const s of data.sessions) {
    const cur = state.current === s.session_id;
    list.append(h("li", {}, h("button", {
      type: "button", "aria-current": cur ? "true" : "false",
      onclick: () => openSession(s.session_id),
    }, s.session_id, h("span", { class: "sub" }, `${s.domain} · ${s.lifecycle_state} · 블러 ${s.privacy_state}`))));
  }
  document.getElementById("no-sessions").hidden = data.sessions.length > 0;
  fillUploadSelects(data);
  if (data.running.length) banner(`실행 중인 작업 ${data.running.length}개: ` + data.running.map((j) => j.title).join(", "));
  else banner("");
}

function fillUploadSelects(data) {
  const dom = document.getElementById("domain-select");
  if (!dom.options.length) for (const d of data.domains) dom.append(h("option", { value: d }, d));
  const sc = document.getElementById("scene-select");
  if (sc.options.length <= 1) for (const s of data.scenes) sc.append(h("option", { value: s }, s));
}

// ── 세션 상세 ──
async function openSession(sid) {
  state.current = sid;
  const box = document.getElementById("session-detail");
  let data;
  try { data = await api("/api/sessions/" + encodeURIComponent(sid)); }
  catch (e) { box.replaceChildren(h("p", { class: "err" }, e.message)); return; }
  const busy = data.jobs.find((j) => j.status === "running") || null;
  fill(box,
    sessionHeader(data),
    busy ? h("p", { class: "next" }, `작업 실행 중: ${busy.title}. 끝나면 화면이 저절로 바뀝니다 (로그는 아래 표).`) : null,
    h("h3", {}, "파이프라인"),
    ...data.steps.map((s, i) => stepCard(data, s, i + 1, busy)),
    resultsCard(data),
    jobsCard(data.jobs),
  );
  for (const b of document.querySelectorAll("#session-list button")) {
    b.setAttribute("aria-current", b.firstChild.textContent === sid ? "true" : "false");
  }
  schedulePoll(data.jobs.some((j) => j.status === "running"));
}

function schedulePoll(active) {
  clearTimeout(state.pollTimer);
  if (active) state.pollTimer = setTimeout(() => { if (state.current) openSession(state.current); loadOverview(); }, 4000);
}

function sessionHeader(data) {
  const s = data.session;
  const warn = [];
  if (data.held) warn.push(h("span", { class: "pill blocked" }, "사고 대응 보류 중"));
  if (data.withdrawn) warn.push(h("span", { class: "pill blocked" }, "사용 중지"));
  return h("div", { class: "card" },
    h("h2", {}, s.session_id, " ", ...warn),
    h("div", { class: "muted small" },
      `도메인 ${s.domain} · 작업자 ${s.worker_id} · 장소 ${s.site_id} · 촬영 ${s.recorded_at.slice(0, 16).replace("T", " ")} UTC · 길이 ${(s.duration_ms / 1000).toFixed(1)}초`),
    h("div", { class: "muted small" },
      `생애주기 ${s.lifecycle_state} · 블러 ${s.privacy_state} · 스트림 ${s.streams.map((x) => x.stream_id).join(", ")}`),
  );
}

function inputFor(name, data) {
  const ov = state.overview || {};
  if (name === "privacy_assignee") {
    const sel = h("select", { name: "assignee", "aria-label": "블러 검수 담당자 (원본 접근 권한자)" });
    for (const r of ov.privacy_reviewers || []) sel.append(h("option", { value: r }, r));
    return h("label", { class: "small" }, "담당자 ", sel);
  }
  if (name === "assignee") {
    return h("label", { class: "small" }, "담당자 ", h("input", { name: "assignee", value: ov.labeler || "", size: 14, "aria-label": "라벨 검수 담당자" }));
  }
  if (name === "tool") {
    const sel = h("select", { name: "tool", "aria-label": "검수 도구" },
      h("option", { value: "both" }, "CVAT + Label Studio"),
      h("option", { value: "label_studio" }, "Label Studio만 (CVAT 꺼짐)"),
      h("option", { value: "cvat" }, "CVAT만"));
    return h("label", { class: "small" }, "도구 ", sel);
  }
  if (name === "scene") {
    const sel = h("select", { name: "scene", "aria-label": "장면" }, h("option", { value: "" }, "올릴 때 고른 값/자동"));
    for (const s of ov.scenes || []) sel.append(h("option", { value: s, selected: data.prefs.scene === s }, s));
    return h("label", { class: "small" }, "장면 ", sel);
  }
  if (name === "quick") {
    return h("label", { class: "small" }, h("input", { type: "checkbox", name: "quick" }), " 빠르게 (전신·깊이 3D 빼기)");
  }
  return null;
}

function stepCard(data, step, n, busy) {
  const inputs = step.inputs.map((x) => inputFor(x, data)).filter(Boolean);
  const runnable = !busy && (step.status === "ready" || step.status === "done");
  const btn = h("button", {
    type: "button", class: "primary", disabled: !runnable,
    "aria-describedby": "why-" + step.id,
    onclick: (ev) => runStep(step, ev.target.closest(".step")),
  }, busy && busy.step === step.id ? "실행 중" : step.status === "done" ? "다시 실행" : "실행");
  const card = h("div", { class: "card step", "data-status": step.status },
    h("div", { class: "num", "aria-hidden": "true" }, step.status === "done" ? "✓" : n),
    h("div", {},
      h("div", {}, h("span", { class: "title" }, step.title), " ", pill(step.status)),
      h("div", { class: "muted small" }, step.help),
      h("div", { class: "why", id: "why-" + step.id }, step.reason || ""),
      inputs.length ? h("div", { class: "inputs" }, inputs) : null,
      step.links.length ? taskLinks(data.tasks, step.links) : null,
    ),
    btn,
  );
  return card;
}

function taskLinks(tasks, stages) {
  const items = tasks.filter((t) => stages.includes(t.stage));
  if (!items.length) return null;
  return h("ul", { class: "links" }, items.map((t) => h("li", {},
    `${t.task_key} (${t.stream_id}, 담당 ${t.assignee || "-"}) `, pill(t.status), " ",
    t.url ? h("a", { href: t.url, target: "_blank", rel: "noopener noreferrer" }, t.tool === "cvat" ? "CVAT에서 열기" : "Label Studio 열기") : null,
  )));
}

async function runStep(step, card) {
  if (step.confirm && !window.confirm(step.confirm)) return;
  const params = {};
  for (const el of card.querySelectorAll("input, select")) {
    params[el.name] = el.type === "checkbox" ? el.checked : el.value;
  }
  try {
    const job = await api(`/api/sessions/${encodeURIComponent(state.current)}/steps/${step.id}`, { method: "POST", json: params });
    banner(`시작함: ${job.title}`);
    openSession(state.current);
  } catch (e) { banner(e.message, true); }
}

// ── 결과 보기 ──
function resultsCard(data) {
  const rows = data.labels.map((r) => h("tr", {}, h("td", {}, r.kind), h("td", {}, r.source), h("td", {}, r.verification_state), h("td", {}, r.count)));
  const table = rows.length
    ? h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "종류"), h("th", {}, "출처"), h("th", {}, "검수 상태"), h("th", {}, "개수"))), h("tbody", {}, rows))
    : h("p", { class: "muted" }, "라벨이 아직 없습니다.");
  const streams = data.session.streams.filter((s) => s.kind === "bodycam" || s.kind === "third_person");
  const sel = h("select", { "aria-label": "미리 볼 스트림" }, streams.map((s) => h("option", { value: s.stream_id }, s.stream_id)));
  const area = h("div", {});
  const btn = h("button", { type: "button", class: "ghost", onclick: () => showPreview(sel.value, area) }, "블러본 위에 라벨 보기");
  return h("div", { class: "card" },
    h("h3", {}, "결과 보기"),
    h("p", { class: "muted small" }, "지금 라벨 수 (화면용 집계). 영상은 승인·렌더된 블러본만 보입니다."),
    table,
    h("div", { class: "row" }, sel, btn),
    area,
  );
}

async function showPreview(stream, area) {
  let data;
  try { data = await api(`/api/sessions/${encodeURIComponent(state.current)}/preview?stream=${encodeURIComponent(stream)}`); }
  catch (e) { area.replaceChildren(h("p", { class: "err" }, e.message)); return; }
  const video = h("video", { controls: true, src: data.url, preload: "metadata", "aria-label": "블러본 영상" });
  const canvas = h("canvas", {});
  const note = h("p", { class: "muted small" }, `트랙 ${data.tracks.length}개 표시` + (data.hidden ? `, ${data.hidden}개 숨김 (신뢰도 낮은 순)` : "") + ". 가까운 앞 키프레임을 그리는 근사 표시입니다.");
  area.replaceChildren(h("div", { class: "player" }, video, canvas), note);
  const draw = () => drawOverlay(video, canvas, data.tracks);
  video.addEventListener("timeupdate", draw);
  video.addEventListener("loadedmetadata", draw);
  video.addEventListener("seeked", draw);
}

function frameAt(frames, tMs) {
  // 시각 순 키프레임에서 tMs 이하의 마지막 것 (500 ms 넘게 지났으면 그리지 않는다)
  let lo = 0, hi = frames.length - 1, best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid][0] <= tMs) { best = mid; lo = mid + 1; } else hi = mid - 1;
  }
  if (best < 0 || tMs - frames[best][0] > 500) return null;
  return frames[best];
}

function drawOverlay(video, canvas, tracks) {
  const w = video.clientWidth, hgt = video.clientHeight;
  if (!video.videoWidth) return;
  canvas.width = w; canvas.height = hgt;
  const sx = w / video.videoWidth, sy = hgt / video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, hgt);
  const css = getComputedStyle(document.documentElement);
  const tMs = video.currentTime * 1000;
  ctx.lineWidth = 2; ctx.font = "12px sans-serif";
  for (const t of tracks) {
    const f = frameAt(t.frames, tMs);
    if (!f) continue;
    const color = t.state === "unreviewed" ? css.getPropertyValue("--warn") : css.getPropertyValue("--ok");
    ctx.strokeStyle = color; ctx.fillStyle = color;
    if (t.type === "box") {
      ctx.strokeRect(f[1] * sx, f[2] * sy, f[3] * sx, f[4] * sy);
      ctx.fillText(t.label, f[1] * sx + 2, Math.max(12, f[2] * sy - 3));
    } else {
      for (const p of f[1]) { ctx.beginPath(); ctx.arc(p[0] * sx, p[1] * sy, 2.5, 0, Math.PI * 2); ctx.fill(); }
    }
  }
}

// ── 작업 ──
function jobsCard(jobs) {
  return h("div", { class: "card" }, h("h3", {}, "이 세션의 작업"), jobsTable(jobs));
}

function jobsTable(jobs) {
  if (!jobs.length) return h("p", { class: "muted" }, "작업이 없습니다.");
  return h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "시작"), h("th", {}, "단계"), h("th", {}, "상태"), h("th", {}, "마지막 오류·다음 할 일"), h("th", {}, ""))),
    h("tbody", {}, jobs.map((j) => h("tr", {},
      h("td", {}, j.created_at.slice(5, 19).replace("T", " ")),
      h("td", {}, j.title, j.session_id ? h("div", { class: "muted small" }, j.session_id) : null),
      h("td", {}, pill(j.status), j.exit_code ? ` (종료 코드 ${j.exit_code})` : ""),
      h("td", {}, j.error_line ? h("div", { class: "err small" }, j.error_line) : null, j.next_action ? h("div", { class: "next small" }, j.next_action) : null),
      h("td", {}, h("button", { type: "button", class: "ghost", onclick: () => showLog(j.job_id) }, "로그"),
        j.step === "ingest" && j.status !== "running" && j.status !== "succeeded"
          ? h("button", { type: "button", class: "ghost", onclick: () => removeUpload(j.session_id) }, "업로드 사본 지우기") : null),
    ))));
}

async function showLog(jobId) {
  const dlg = document.getElementById("log-dialog");
  const render = async () => {
    const j = await api("/api/jobs/" + encodeURIComponent(jobId));
    document.getElementById("log-title").textContent = `${j.title} — ${STATUS_KO[j.status] || j.status}`;
    fill(document.getElementById("log-meta"),
      h("div", { class: "muted small" }, j.commands.join("  →  ")),
      j.error_line ? h("div", { class: "err" }, "마지막 오류: ", j.error_line) : null,
      j.next_action ? h("div", { class: "next" }, "다음 할 일: ", j.next_action) : null,
      (j.warnings || []).length ? h("ul", { class: "small" }, j.warnings.map((w) => h("li", {}, w))) : null,
    );
    const pre = document.getElementById("log-body");
    pre.textContent = (j.log || []).join("\n") || "(아직 출력 없음)";
    pre.scrollTop = pre.scrollHeight;
    return j.status === "running";
  };
  dlg.showModal();
  let running = await render();
  while (running && dlg.open) {
    await new Promise((r) => setTimeout(r, 2000));
    if (!dlg.open) break;
    running = await render();
  }
}

async function removeUpload(sid) {
  if (!window.confirm(`${sid}: 올린 사본(작업 디렉터리)을 지웁니다. 원본 저장소에 수집된 것은 그대로입니다.`)) return;
  try {
    const r = await api("/api/uploads/" + encodeURIComponent(sid), { method: "DELETE" });
    banner(r.removed ? "지웠습니다. 다시 올릴 수 있습니다." : "지울 사본이 없습니다.");
  } catch (e) { banner(e.message, true); }
}

async function loadJobs() {
  const box = document.getElementById("jobs");
  try { box.replaceChildren(jobsTable(await api("/api/jobs"))); }
  catch (e) { box.replaceChildren(h("p", { class: "err" }, e.message)); }
}

// ── 업로드 ──
function setupUpload() {
  const form = document.getElementById("upload-form");
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const fd = new FormData(form);
    const bar = document.getElementById("upload-progress");
    bar.hidden = false; bar.value = 0;
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.setRequestHeader("X-Console-Token", TOKEN);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) bar.value = (e.loaded / e.total) * 100; };
    xhr.onload = async () => {
      bar.hidden = true;
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
      if (xhr.status >= 300) { banner((data && data.detail) || "올리기 실패", true); return; }
      banner(`올림: ${data.session_id}. 준비·수집을 시작했습니다 (작업 기록에서 진행을 봅니다).`);
      form.reset();
      showTab("jobs");
    };
    xhr.onerror = () => { bar.hidden = true; banner("올리기 중 연결이 끊겼습니다", true); };
    xhr.send(fd);
  });
}

// ── 학습·성과 ──
function lineChart(points, opts) {
  // points: [{x: 이름, y: 값|null, ok: true|false|null}], 인라인 SVG (외부 라이브러리 없음)
  const W = 640, H = 220, L = 46, R = 12, T = 14, B = 46;
  const ys = points.map((p) => p.y).filter((y) => y !== null && y !== undefined);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "chart");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", opts.label);
  const el = (tag, attrs, text) => {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    svg.append(e);
    return e;
  };
  if (!ys.length) { el("text", { x: 10, y: 30 }, "값 없음"); return svg; }
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (opts.unit01) { lo = Math.min(lo, 0); hi = Math.max(hi, 1); }
  if (hi === lo) { hi += 1; lo -= 1; }
  const px = (i) => L + (points.length === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (points.length - 1));
  const py = (y) => T + ((hi - y) * (H - T - B)) / (hi - lo);
  el("line", { x1: L, y1: H - B, x2: W - R, y2: H - B, class: "axis" });
  el("line", { x1: L, y1: T, x2: L, y2: H - B, class: "axis" });
  for (const v of [lo, (lo + hi) / 2, hi]) el("text", { x: 4, y: py(v) + 4 }, v.toFixed(2));
  const path = points.map((p, i) => (p.y === null || p.y === undefined ? null : `${px(i)},${py(p.y)}`)).filter(Boolean);
  if (path.length > 1) el("polyline", { points: path.join(" "), class: "line" });
  points.forEach((p, i) => {
    if (p.y === null || p.y === undefined) return;
    const c = el("circle", { cx: px(i), cy: py(p.y), r: 5, class: p.ok === true ? "pt-pass" : p.ok === false ? "pt-fail" : "pt-none" });
    const tip = document.createElementNS("http://www.w3.org/2000/svg", "title");
    tip.textContent = `${p.x}: ${p.y.toFixed(4)}`;
    c.append(tip);
    el("text", { x: px(i) - 18, y: H - B + 16 }, String(p.x).slice(0, 12));
  });
  return svg;
}

function roundsByTask(rounds) {
  const out = {};
  for (const r of rounds) (out[r.task] = out[r.task] || []).push(r);
  return out;
}

function taskSection(task, items, trainers) {
  const reported = items.filter((r) => r.report);
  const metricNames = reported.length ? Object.keys(reported[reported.length - 1].report.metrics) : [];
  const metricSel = h("select", { "aria-label": `${task} 지표` }, metricNames.map((m) => h("option", { value: m }, m)));
  const chartBox = h("div", {});
  const draw = () => {
    const m = metricSel.value;
    chartBox.replaceChildren(lineChart(items.map((r, i) => ({
      x: `${i + 1}회 ${r.status}`, y: r.report && m in r.report.metrics ? r.report.metrics[m] : null,
      ok: r.report ? r.report.gate_passed : null,
    })), { label: `${task} 회차별 골든셋 ${m}`, unit01: true }));
  };
  metricSel.addEventListener("change", draw);
  const policyTrainer = trainers[task];
  const badges = [];
  if (policyTrainer && policyTrainer.startsWith("oracle")) badges.push(h("span", { class: "badge-test" }, "시험용 학습기"));
  const table = h("table", {},
    h("thead", {}, h("tr", {}, ["회차", "모델 버전", "학습기", "상태", "예제", "게이트 이유", ""].map((x) => h("th", {}, x)))),
    h("tbody", {}, items.map((r, i) => h("tr", {},
      h("td", {}, i + 1),
      h("td", {}, h("code", {}, r.model_version)),
      h("td", {}, r.trainer, r.test_trainer ? h("span", { class: "badge-test" }, "시험용") : null),
      h("td", {}, pill(r.status)),
      h("td", {}, r.train_examples),
      h("td", { class: "small" }, r.report ? (r.report.gate_reasons.join(" / ") || "-") : "리포트 없음"),
      h("td", {}, r.status === "passed" ? h("button", { type: "button", class: "primary", onclick: () => runGlobal("train_approve", { model_version: r.model_version }, "게이트를 통과한 이 모델을 배포합니다. 계속할까요?") }, "배포 승인") : null),
    ))));
  const sec = h("div", { class: "card" },
    h("h3", {}, `과제 ${task}`, " ", h("span", { class: "muted small" }, `정책 학습기: ${policyTrainer || "-"}`), ...badges),
    metricNames.length ? h("label", { class: "small" }, "지표 ", metricSel) : h("p", { class: "muted small" }, "평가 리포트가 있는 회차가 없습니다."),
    chartBox, table);
  if (metricNames.length) draw();
  return sec;
}

function weeklySection(weekly) {
  if (!weekly) return h("div", { class: "card" }, h("h3", {}, "수정률 추이"), h("p", { class: "muted" }, "아직 주간 지표를 계산하지 않았습니다. 아래 '주간 운영 지표'를 누르세요."));
  return h("div", { class: "card" },
    h("h3", {}, "수정률 추이 (주별)"),
    h("p", { class: "muted small" }, `수정률 = (고침+지움+추가) / (승인+고침+지움+추가). 낮아질수록 AI 초안을 덜 고친다는 뜻입니다. 자료: ${weekly.file}`),
    lineChart(weekly.weeks.map((w) => ({ x: w.week, y: w.correction_rate, ok: null })), { label: "주별 수정률", unit01: true }),
    weekly.alerts.length ? h("ul", {}, weekly.alerts.map((a) => h("li", { class: "small" }, typeof a === "string" ? a : JSON.stringify(a)))) : null,
  );
}

function globalForm(title, stepId, fields, note) {
  const inputs = fields.map((f) => f.options
    ? h("label", { class: "small" }, f.label, " ", h("select", { name: f.name }, f.options.map((o) => h("option", { value: o }, o || "(없음)"))))
    : f.check
      ? h("label", { class: "small" }, h("input", { type: "checkbox", name: f.name }), " ", f.label)
      : h("label", { class: "small" }, f.label, " ", h("input", { name: f.name, placeholder: f.placeholder || "", size: f.size || 16 })));
  const card = h("div", { class: "card" }, h("h3", {}, title), note ? h("p", { class: "muted small" }, note) : null, h("div", { class: "inputs row" }, inputs,
    h("button", { type: "button", class: "primary", onclick: () => {
      const p = {};
      for (const el of card.querySelectorAll("input, select")) p[el.name] = el.type === "checkbox" ? el.checked : el.value;
      runGlobal(stepId, p);
    } }, "실행")));
  return card;
}

async function runGlobal(stepId, params, confirmText) {
  if (confirmText && !window.confirm(confirmText)) return;
  try {
    const job = await api("/api/global/" + stepId, { method: "POST", json: params });
    banner(`시작함: ${job.title}`);
    setTimeout(loadTraining, 1500);
  } catch (e) { banner(e.message, true); }
}

async function loadTraining() {
  const box = document.getElementById("training");
  let data;
  try { data = await api("/api/training"); }
  catch (e) { box.replaceChildren(h("p", { class: "err" }, e.message)); return; }
  const tasks = Object.keys(data.trainers);
  const byTask = roundsByTask(data.rounds);
  const anyTest = Object.values(data.trainers).some((t) => t.startsWith("oracle"));
  fill(box,
    anyTest ? h("p", { class: "next" }, "주의: 지금 정책 학습기 중 oracle-stub(시험용 학습기)이 있습니다. 그 과제의 지표는 실제 모델 성능이 아닙니다.") : null,
    weeklySection(data.weekly),
    h("div", { class: "grid2" },
      globalForm("데이터셋 버전 만들기", "dataset_build", [
        { name: "version_id", label: "버전 ID", placeholder: "예: ds-2026-10-27" },
        { name: "golden", label: "골든셋", options: ["", ...data.golden_sets] },
        { name: "domain", label: "도메인", options: ["", "cleaning", "caregiving", "nursing"] },
        { name: "parent", label: "이전 버전", placeholder: "(선택)" },
      ], "검수 완료(human_verified) 세션으로 작업자·장소가 겹치지 않게 나눕니다."),
      globalForm("재학습 (골든셋 게이트)", "train_run", [
        { name: "task", label: "과제", options: tasks },
        { name: "dataset_version", label: "데이터셋 버전" },
        { name: "baseline_versions", label: "처음 비교할 기본 모델 버전", placeholder: "(첫 배포 전, 공백으로 구분)", size: 24 },
        { name: "force", label: "누적 조건 무시", check: true },
      ], "학습 → 골든셋 평가 → 게이트. 통과하면 정책대로 자동 배포하거나 승인 대기로 둡니다."),
      globalForm("골든셋 평가", "eval_golden", [
        { name: "golden", label: "골든셋", options: data.golden_sets },
        { name: "models", label: "모델 (<과제>=<버전>)", size: 26 },
        { name: "baseline", label: "비교 (선택)", size: 22 },
      ], "DB에 있는 예측만 평가합니다 (쓰기 없음). 리포트는 작업 디렉터리 reports/에 남습니다."),
      globalForm("검수 품질 · 주간 운영 지표", "ops_weekly", [{ name: "weeks", label: "주 수", placeholder: "8", size: 4 }],
        "주간 지표(수정률 그래프의 자료)를 다시 계산합니다. 검수 품질은 작업 기록의 로그로 봅니다."),
    ),
    h("div", { class: "row" }, h("button", { type: "button", class: "ghost", onclick: () => runGlobal("review_quality", {}) }, "검수 품질 계산")),
    h("h3", {}, "회차별 골든셋 지표"),
    ...Object.keys(byTask).map((t) => taskSection(t, byTask[t], data.trainers)),
    Object.keys(byTask).length ? null : h("p", { class: "muted" }, "아직 재학습 회차가 없습니다."),
    h("h3", {}, "세션 밖 작업"), jobsTable(data.jobs),
  );
}

// ── 가중치 가져오기 ──
async function loadPretrain() {
  const box = document.getElementById("pretrain");
  let data;
  try { data = await api("/api/pretrain"); }
  catch (e) { box.replaceChildren(h("p", { class: "err" }, e.message)); return; }
  const form = h("form", { class: "form" },
    h("label", {}, "ONNX 가중치 (objects_pretrain_yolox_s.onnx)", h("input", { type: "file", name: "onnx", accept: ".onnx", required: true })),
    h("label", {}, "model_manifest.json", h("input", { type: "file", name: "model_manifest", accept: ".json", required: true })),
    h("label", {}, "data_manifest.json", h("input", { type: "file", name: "data_manifest", accept: ".json", required: true })),
    h("div", { class: "row" }, h("button", { type: "submit", class: "primary" }, "검사하기")));
  const result = h("div", {});
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    try {
      const plan = await api("/api/pretrain/check", { method: "POST", body: new FormData(form) });
      result.replaceChildren(planView(plan, data));
    } catch (e) { result.replaceChildren(h("p", { class: "err" }, e.message)); }
  });
  fill(box,
    h("p", { class: "muted" }, "Colab·Kaggle에서 공개 데이터만으로 학습한 결과(scripts/pretrain)를 올립니다. 우리 데이터(블러본·라벨)를 밖으로 보내는 기능은 없습니다 (2단계 다듬기는 사내 GPU에서만)."),
    form, result,
    gateSection(data),
    h("h3", {}, "확인한 가져오기"),
    data.imports.length ? h("table", {}, h("thead", {}, h("tr", {}, ["시각", "모델 이름", "분류", "위치"].map((x) => h("th", {}, x)))),
      h("tbody", {}, data.imports.map((r) => h("tr", {}, h("td", {}, r.at.slice(0, 16)), h("td", {}, h("code", {}, r.model_name)), h("td", {}, r.commercial), h("td", {}, r.dest)))))
      : h("p", { class: "muted" }, "아직 없습니다."),
  );
}

function planView(plan, data) {
  const checks = h("ul", {}, plan.checks.map((c) => h("li", { class: c.ok ? "check-ok" : "check-bad" }, (c.ok ? "통과 " : "실패 ") + c.name, h("span", { class: "muted small" }, " — " + c.detail))));
  const parts = [h("h3", {}, "검사 결과 ", h("span", { class: "pill " + (plan.ok ? "done" : "failed") }, plan.ok ? "검사 통과" : "검사 실패")), checks];
  for (const p of plan.license_problems) parts.push(h("p", { class: "err" }, "라이선스: " + p));
  if (plan.model_name) {
    parts.push(
      h("p", {}, "모델 이름 ", h("code", {}, plan.model_name), ` · 상업 사용 분류 ${plan.commercial} · 둘 곳 ${plan.dest}`),
      h("h3", {}, "config/models.yaml에 넣을 내용 (미리보기)"),
      h("p", { class: "muted small" }, "콘솔은 Git이 관리하는 설정 파일을 직접 고치지 않습니다. 확인하면 패치 파일을 만들고 명령을 안내합니다."),
      h("pre", { tabindex: 0 }, plan.models_patch || "(바꿀 것 없음)"),
      ...plan.notes.map((n) => h("p", { class: "next small" }, n)),
    );
  }
  const out = h("div", {});
  if (plan.ok) {
    parts.push(h("div", { class: "row" }, h("button", { type: "button", class: "primary", onclick: async () => {
      if (!window.confirm("가중치를 data/models 아래에 두고 가져오기 기록을 남깁니다. 계속할까요?")) return;
      try {
        const rec = await api(`/api/pretrain/${encodeURIComponent(plan.import_id)}/confirm`, { method: "POST" });
        out.replaceChildren(h("p", {}, "기록했습니다. 터미널에서 다음을 실행하세요:"), h("pre", {}, rec.next.join("\n")), compareForm(plan, data));
      } catch (e) { out.replaceChildren(h("p", { class: "err" }, e.message)); }
    } }, "확인하고 기록")));
  }
  return h("div", { class: "card" }, ...parts, out);
}

function compareForm(plan, data) {
  if (!data.compare_trainer) {
    return h("p", { class: "next" }, "골든셋 비교: 외부 가중치를 예측을 DB에 쓰지 않고 골든셋으로 평가하는 학습기가 아직 연결되지 않았습니다 (config/console.yaml pretrain_import.compare.trainer, F-COE-74).");
  }
  const ds = h("input", { name: "dataset_version", placeholder: "데이터셋 버전" });
  const base = h("input", { name: "baseline_versions", placeholder: "지금 도구 단계 모델 버전 (첫 비교)", size: 30 });
  return h("div", { class: "card" }, h("h3", {}, "골든셋 비교 (지금 OWLv2 도구 단계와)"),
    h("p", { class: "muted small" }, `학습 루프 게이트 경로(dlp train run objects … --trainer ${data.compare_trainer})를 씁니다. 예측은 DB에 쓰지 않습니다. objects 과제가 deploy: auto면 통과 즉시 배포됩니다.`),
    h("div", { class: "row" }, ds, base, h("button", { type: "button", class: "primary", onclick: async () => {
      try {
        const job = await api(`/api/pretrain/${encodeURIComponent(plan.import_id)}/compare`, { method: "POST", json: { dataset_version: ds.value, baseline_versions: base.value } });
        banner(`시작함: ${job.title}`);
      } catch (e) { banner(e.message, true); }
    } }, "비교 실행")));
}

function gateSection(data) {
  const last = data.imports[0];
  if (!last) return null;
  if (!data.gate_passed) {
    return h("div", { class: "card" }, h("h3", {}, "closed_tools 단계 켜기"),
      h("p", { class: "muted" }, "게이트를 통과한 비교 결과가 없어 켜지 않습니다. 골든셋 비교에서 게이트를 통과하면 켜는 방법이 여기 나옵니다."));
  }
  return h("div", { class: "card" }, h("h3", {}, "closed_tools 단계 켜기 (게이트 통과)"),
    h("p", {}, "config/policies/prelabel.yaml에서 다음처럼 바꾸고 Git으로 검토한 뒤 다음 AI 라벨부터 씁니다:"),
    h("pre", {}, `models:\n  closed_vocab_tools: ${last.model_name}`),
    h("p", { class: "muted small" }, "켜면 같은 클래스는 OWLv2 질의에서 빠지고 도구 단계가 다시 돕니다 (검수 전 결과만 바뀝니다)."));
}

// ── 시작 ──
document.getElementById("mode").addEventListener("click", () => setMode(document.documentElement.dataset.mode === "dark" ? "light" : "dark"));
document.getElementById("refresh").addEventListener("click", () => loadOverview().catch((e) => banner(e.message, true)));
document.getElementById("log-close").addEventListener("click", () => document.getElementById("log-dialog").close());
for (const tab of document.querySelectorAll('[role="tab"]')) tab.addEventListener("click", () => showTab(tab.id.slice(4)));
setupUpload();
if (!TOKEN) banner("콘솔 토큰이 없습니다. `dlp console serve`가 찍은 주소(#token=…)로 여세요.", true);
else loadOverview().catch((e) => banner(e.message, true));
