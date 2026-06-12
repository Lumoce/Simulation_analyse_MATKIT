const state = {
  defaults: {},
  browserTarget: null,
  browserPath: "",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function setStatus(text, tone = "ready") {
  const node = $("#serverStatus");
  node.textContent = text;
  node.dataset.tone = tone;
}

function currentMode() {
  return $("#energyType").value;
}

function isVisibleForMode(element, mode) {
  const show = element.dataset.show;
  if (!show) return true;
  return show.split(/\s+/).includes(mode);
}

function syncMode() {
  const mode = currentMode();
  $$(".form-section").forEach((section) => {
    section.classList.toggle("hidden", !isVisibleForMode(section, mode));
  });
  $$(".mode-note").forEach((note) => {
    note.classList.toggle("hidden", note.dataset.mode !== mode);
  });
  const labels = {
    surface: {
      initial: "初态目录（可空）",
      final: "表面终态/结果目录",
      hint: "表面能只需要终态目录和单质参考库；初态目录可以留空。若终态目录本身是一个计算结果则单个计算，若其中包含多个 task 计算目录则自动批量计算。",
      finalPlaceholder: "/path/to/surface_final_or_results_directory",
      initialPlaceholder: "可留空",
    },
    adsorption: {
      initial: "吸附前初态目录",
      final: "吸附后终态目录",
      hint: "吸附能 = 终态能量 - 初态能量 - n * 吸附物能量。初态可以直接选择前一步计算的终态目录。",
      finalPlaceholder: "/path/to/slab_plus_adsorbate",
      initialPlaceholder: "/path/to/clean_slab",
    },
    doping: {
      initial: "缺陷前初态目录",
      final: "缺陷/掺杂后终态目录",
      hint: "缺陷/掺杂形成能使用显式选择的初态和终态目录，不需要复制同一结构到多个分类文件夹。",
      finalPlaceholder: "/path/to/doped_or_defective_state",
      initialPlaceholder: "/path/to/pristine_state",
    },
  };
  const config = labels[mode] || labels.surface;
  $("#initialStateLabel").textContent = config.initial;
  $("#finalStateLabel").textContent = config.final;
  $("#stateHint").textContent = config.hint;
  $("#initialState").placeholder = config.initialPlaceholder;
  $("#finalState").placeholder = config.finalPlaceholder;
}

async function apiGet(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!data.ok) throw new Error(data.error || "请求失败");
  return data;
}

async function apiPost(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!data.ok) throw new Error(data.error || "计算失败");
  return data;
}

function setDefaultPaths() {
  if (!state.defaults.projectRoot) return;
  $("#referenceDb").value = state.defaults.defaultReferenceDb || "";
  $("#csvPath").value = `${state.defaults.defaultOutputDir}/energy_ui_results.csv`;
  $("#jsonPath").value = `${state.defaults.defaultOutputDir}/energy_ui_results.json`;
}

async function loadDefaults() {
  try {
    state.defaults = await apiGet("/api/defaults");
    setDefaultPaths();
  } catch (error) {
    setStatus("Defaults unavailable", "error");
  }
}

function formPayload() {
  const mode = currentMode();
  const payload = {
    energyType: mode,
    initialPath: $("#initialState").value,
    finalPath: $("#finalState").value,
    referenceDb: $("#referenceDb").value,
    nSurfaces: $("#nSurfaces").value,
    csvPath: $("#csvPath").value,
    jsonPath: $("#jsonPath").value,
  };

  if (mode === "adsorption") {
    payload.adsorbatePath = $("#adsorbate").value;
    payload.nAdsorbate = $("#nAdsorbate").value;
  } else if (mode === "doping") {
    payload.nDopant = $("#nDopant").value;
    payload.muDopant = $("#muDopant").value;
    payload.muHost = $("#muHost").value;
    payload.chargeState = $("#chargeState").value;
    payload.efermi = $("#efermi").value;
    payload.correction = $("#correction").value;
  }
  return payload;
}

function formatCell(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    return Math.abs(value) >= 1000 || Math.abs(value) < 0.001 && value !== 0
      ? value.toExponential(4)
      : value.toFixed(6).replace(/\.?0+$/, "");
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function preferredColumns(rows) {
  const available = new Set();
  rows.forEach((row) => Object.keys(row).forEach((key) => available.add(key)));
  const preferred = [
    "label",
    "task_number",
    "task_suffix",
    "status",
    "surface_energy_eV_per_surface",
    "adsorption_energy_eV",
    "formation_energy_eV",
    "excess_energy_eV",
    "energy_diff",
    "reference_energies",
    "formula_terms",
    "E_slab",
    "E_slab_ads",
    "E_doped",
    "reference_energy_eV",
    "n_surfaces",
    "error",
  ];
  const columns = preferred.filter((key) => available.has(key));
  Array.from(available).sort().forEach((key) => {
    if (!columns.includes(key) && columns.length < 14) columns.push(key);
  });
  return columns;
}

function renderResults(data) {
  const rows = data.results || [];
  const thead = $("#resultsTable thead");
  const tbody = $("#resultsTable tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  const summary = data.summary || {};
  $("#summaryText").textContent = `total ${summary.total ?? rows.length} | ok ${summary.ok ?? 0} | errors ${summary.errors ?? 0}`;

  const saved = data.saved || {};
  const savedNode = $("#savedPaths");
  savedNode.innerHTML = "";
  Object.entries(saved).forEach(([kind, path]) => {
    const item = document.createElement("span");
    item.textContent = `${kind}: ${path}`;
    savedNode.appendChild(item);
  });

  if (!rows.length) {
    tbody.innerHTML = "<tr><td class=\"empty\">暂无结果</td></tr>";
    return;
  }

  const columns = preferredColumns(rows);
  const headerRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((column) => {
      const td = document.createElement("td");
      td.textContent = formatCell(row[column]);
      if (column === "status" && row[column] !== "ok") td.classList.add("error-text");
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

async function runCalculation(event) {
  event.preventDefault();
  setStatus("Running", "busy");
  $("#summaryText").textContent = "计算中";
  try {
    const data = await apiPost("/api/calculate", formPayload());
    renderResults(data);
    setStatus("Done", "ready");
  } catch (error) {
    setStatus("Error", "error");
    $("#summaryText").textContent = error.message;
    $("#savedPaths").innerHTML = "";
    $("#resultsTable thead").innerHTML = "";
    $("#resultsTable tbody").innerHTML = `<tr><td class="empty error-text">${error.message}</td></tr>`;
  }
}

function itemElement(entry) {
  const row = document.createElement("div");
  row.className = "browser-item";

  const kind = document.createElement("span");
  kind.className = "kind";
  kind.textContent = entry.isDir ? "dir" : "file";

  const name = document.createElement("span");
  name.className = "name";
  name.textContent = entry.name;
  name.title = entry.path;
  if (entry.isDir) {
    name.addEventListener("dblclick", (event) => {
      event.stopPropagation();
      loadBrowser(entry.path);
    });
  }

  const button = document.createElement("button");
  button.type = "button";
  button.textContent = entry.isDir ? "打开" : "选择";
  button.addEventListener("click", () => {
    if (entry.isDir) {
      loadBrowser(entry.path);
    } else {
      choosePath(entry.path);
    }
  });

  row.append(kind, name, button);
  row.addEventListener("dblclick", () => {
    if (entry.isDir) loadBrowser(entry.path);
    else choosePath(entry.path);
  });
  return row;
}

function choosePath(path) {
  const target = state.browserTarget ? document.getElementById(state.browserTarget) : null;
  if (target) target.value = path || state.browserPath;
  $("#browserDialog").close();
}

async function loadBrowser(path) {
  try {
    const data = await apiGet(`/api/browse?path=${encodeURIComponent(path || "")}`);
    state.browserPath = data.path;
    $("#browserPath").value = data.path;

    const quick = $("#quickRoots");
    quick.innerHTML = "";
    (data.quickRoots || []).forEach((root) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = root;
      button.addEventListener("click", () => loadBrowser(root));
      quick.appendChild(button);
    });

    const list = $("#browserList");
    list.innerHTML = "";
    if (data.parent && data.parent !== data.path) {
      list.appendChild(itemElement({name: "..", path: data.parent, isDir: true}));
    }
    data.entries.forEach((entry) => list.appendChild(itemElement(entry)));
  } catch (error) {
    $("#browserList").innerHTML = `<div class="browser-item error-text">${error.message}</div>`;
  }
}

function openBrowser(targetId) {
  state.browserTarget = targetId;
  const current = document.getElementById(targetId)?.value || state.defaults.projectRoot || "";
  $("#browserDialog").showModal();
  loadBrowser(current);
}

function bindEvents() {
  $("#energyType").addEventListener("change", syncMode);
  $("#energyForm").addEventListener("submit", runCalculation);
  $("#resetForm").addEventListener("click", () => {
    $("#energyForm").reset();
    setDefaultPaths();
    syncMode();
  });

  $$(".browse-btn").forEach((button) => {
    button.addEventListener("click", () => openBrowser(button.dataset.target));
  });

  $("#closeBrowser").addEventListener("click", () => $("#browserDialog").close());
  $("#goPath").addEventListener("click", () => loadBrowser($("#browserPath").value));
  $("#choosePath").addEventListener("click", () => choosePath($("#browserPath").value));
  $("#browserPath").addEventListener("keydown", (event) => {
    if (event.key === "Enter") loadBrowser($("#browserPath").value);
  });
}

bindEvents();
syncMode();
loadDefaults();
