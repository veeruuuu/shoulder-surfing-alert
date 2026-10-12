const stateBadge = document.getElementById("state-badge");
const statusLine = document.getElementById("status-line");
const actionMsg = document.getElementById("action-msg");

function setBadge(state) {
  if (!stateBadge) return;
  stateBadge.textContent = state;
  stateBadge.className = "badge " + state.toLowerCase();
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    setBadge(data.paused ? "PAUSED" : data.state);

    const camDot = document.getElementById("camera-indicator");
    const camLabel = document.getElementById("camera-label");
    if (camDot && camLabel) {
      camDot.className = "cam-dot " + (data.camera_active ? "active" : "inactive");
      camLabel.textContent = "camera: " + (data.camera_active ? "active" : "inactive");
    }

    if (statusLine) {
      statusLine.textContent =
        `fps: ${data.fps} | faces: ${data.face_count} | calibrated: ${data.calibrated} | paused: ${data.paused}`;
    }

    // Camera permission/availability errors (§15 B7) surfaced prominently.
    if (data.camera_error) {
      if (actionMsg) actionMsg.textContent = "Camera error: " + data.camera_error;
    } else if (actionMsg && actionMsg.textContent.startsWith("Camera error:")) {
      actionMsg.textContent = "";
    }
  } catch (e) {
    if (statusLine) statusLine.textContent = "status unavailable";
  }
}

// --- Live page: pause/resume ---
const btnPause = document.getElementById("btn-pause");
const btnResume = document.getElementById("btn-resume");
if (btnPause) {
  btnPause.addEventListener("click", async () => {
    await fetch("/api/engine/pause", { method: "POST" });
    refreshStatus();
  });
}
if (btnResume) {
  btnResume.addEventListener("click", async () => {
    await fetch("/api/engine/resume", { method: "POST" });
    refreshStatus();
  });
}

// --- Calibration page ---
const btnCalibrate = document.getElementById("btn-calibrate");
if (btnCalibrate) {
  btnCalibrate.addEventListener("click", async () => {
    if (actionMsg) actionMsg.textContent = "Calibrating... sit alone in frame.";
    const res = await fetch("/api/calibrate", { method: "POST" });
    const data = await res.json();
    if (actionMsg) {
      actionMsg.textContent = data.success ? "Calibration successful." : `Calibration failed: ${data.error}`;
    }
    refreshStatus();
  });
}

// --- Trusted faces page ---
const trustedTableBody = document.querySelector("#trusted-table tbody");
const btnEnroll = document.getElementById("btn-enroll");

async function refreshTrusted() {
  if (!trustedTableBody) return;
  try {
    const res = await fetch("/api/trusted");
    const faces = await res.json();
    trustedTableBody.innerHTML = "";
    for (const f of faces) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${f.name}</td>
        <td>${f.is_owner ? "yes" : "no"}</td>
        <td>${f.is_owner ? "" : `<button data-id="${f.id}" class="del-trusted">Delete</button>`}</td>
      `;
      trustedTableBody.appendChild(tr);
    }
    document.querySelectorAll(".del-trusted").forEach((btn) => {
      btn.addEventListener("click", async () => {
        await fetch(`/api/trusted/${btn.dataset.id}`, { method: "DELETE" });
        refreshTrusted();
      });
    });
  } catch (e) {
    trustedTableBody.innerHTML = "<tr><td colspan='3'>unavailable</td></tr>";
  }
}

if (btnEnroll) {
  btnEnroll.addEventListener("click", async () => {
    const nameInput = document.getElementById("enroll-name");
    const name = nameInput.value.trim();
    if (!name) {
      if (actionMsg) actionMsg.textContent = "Enter a name first.";
      return;
    }
    if (actionMsg) actionMsg.textContent = "Enrolling... sit alone in frame.";
    const res = await fetch("/api/trusted", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();
    if (actionMsg) actionMsg.textContent = data.success ? "Enrolled." : `Enroll failed: ${data.error}`;
    nameInput.value = "";
    refreshTrusted();
  });
}

// --- Intruder log page ---
const intrudersTableBody = document.querySelector("#intruders-table tbody");

async function fetchIntruders() {
  try {
    const res = await fetch("/api/intruders");
    return await res.json();
  } catch (e) {
    return [];
  }
}

async function refreshIntruders() {
  if (!intrudersTableBody) return;
  const events = await fetchIntruders();
  intrudersTableBody.innerHTML = "";
  for (const ev of events) {
    const when = new Date(ev.started_at * 1000).toLocaleString();
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${when}</td>
      <td>${ev.duration_s !== null ? ev.duration_s.toFixed(1) : "-"}</td>
      <td>${ev.snapshot_path ? `<a href="/api/intruders/${ev.id}/snapshot" target="_blank">view</a>` : "-"}</td>
      <td><button data-id="${ev.id}" class="del-intruder">Delete</button></td>
    `;
    intrudersTableBody.appendChild(tr);
  }
  document.querySelectorAll(".del-intruder").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await fetch(`/api/intruders/${btn.dataset.id}`, { method: "DELETE" });
      refreshIntruders();
    });
  });
}

// --- Settings & Stats page ---
const statsLine = document.getElementById("stats-line");
const settingsTextarea = document.getElementById("settings-json");
const btnSaveSettings = document.getElementById("btn-save-settings");
const settingsMsg = document.getElementById("settings-msg");
const statsChartCanvas = document.getElementById("stats-chart");
let statsChart = null;

async function refreshStats() {
  if (!statsLine) return;
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    statsLine.textContent =
      `total events: ${data.total_events} | avg duration: ${data.avg_duration_s} s`;
  } catch (e) {
    statsLine.textContent = "stats unavailable";
  }
}

async function refreshStatsChart() {
  if (!statsChartCanvas || typeof Chart === "undefined") return;
  const events = await fetchIntruders();
  const counts = {};
  for (const ev of events) {
    const day = new Date(ev.started_at * 1000).toLocaleDateString();
    counts[day] = (counts[day] || 0) + 1;
  }
  const labels = Object.keys(counts);
  const values = Object.values(counts);

  if (statsChart) {
    statsChart.data.labels = labels;
    statsChart.data.datasets[0].data = values;
    statsChart.update();
    return;
  }
  statsChart = new Chart(statsChartCanvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Peeker events per day", data: values, backgroundColor: "#2a63d6" }],
    },
    options: {
      scales: {
        x: { ticks: { color: "#a7acb8" } },
        y: { ticks: { color: "#a7acb8" }, beginAtZero: true },
      },
      plugins: { legend: { labels: { color: "#a7acb8" } } },
    },
  });
}

async function loadSettings() {
  if (!settingsTextarea) return;
  try {
    const res = await fetch("/api/settings");
    const data = await res.json();
    settingsTextarea.value = JSON.stringify(data, null, 2);
  } catch (e) {
    settingsTextarea.value = "{}";
  }
}

if (btnSaveSettings) {
  btnSaveSettings.addEventListener("click", async () => {
    let parsed;
    try {
      parsed = JSON.parse(settingsTextarea.value);
    } catch (e) {
      settingsMsg.textContent = "Invalid JSON: " + e.message;
      return;
    }
    settingsMsg.textContent = "Saving...";
    const res = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parsed),
    });
    const data = await res.json();
    settingsMsg.textContent = data.success
      ? "Saved. Changes apply automatically within a few seconds."
      : `Failed: ${data.error || "unknown error"}`;
  });
}

// --- SSE: live state badge on every page ---
const sse = new EventSource("/api/events");
sse.onmessage = (e) => {
  const data = JSON.parse(e.data);
  setBadge(data.state);
};

// --- Init (guarded functions no-op on pages without their elements) ---
refreshStatus();
refreshTrusted();
refreshIntruders();
refreshStats();
refreshStatsChart();
loadSettings();

setInterval(refreshStatus, 2000);
setInterval(refreshTrusted, 5000);
setInterval(refreshIntruders, 5000);
setInterval(refreshStats, 5000);
