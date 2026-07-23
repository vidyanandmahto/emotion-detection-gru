/**
 * script.js
 * ---------
 * Handles the interactive prediction UI: input validation, AJAX calls to
 * the /predict endpoint, animated result rendering, and a local prediction
 * history (kept in memory + localStorage for this browser session).
 */

(function () {
  "use strict";

  const textInput = document.getElementById("text-input");
  const charCount = document.getElementById("char-count");
  const predictBtn = document.getElementById("predict-btn");
  const clearBtn = document.getElementById("clear-btn");
  const warning = document.getElementById("warning");
  const resultCard = document.getElementById("result-card");
  const resultEmoji = document.getElementById("result-emoji");
  const resultEmotion = document.getElementById("result-emotion");
  const confidenceValue = document.getElementById("confidence-value");
  const ringFg = document.getElementById("ring-fg");
  const probBars = document.getElementById("prob-bars");
  const historyList = document.getElementById("history-list");
  const resetHistoryBtn = document.getElementById("reset-history");
  const examples = document.getElementById("examples");

  const RING_CIRCUMFERENCE = 2 * Math.PI * 52; // matches r=52 in SVG

  const EMOTION_COLORS = {
    joy: "#FFD166",
    sadness: "#4C7DF0",
    anger: "#EF476F",
    fear: "#8D5BC7",
    love: "#FF6B9D",
    surprise: "#06D6A0",
  };

  const HISTORY_KEY = "emotionlens_history";
  let history = [];

  // Guard: only run on pages that actually have the widget (index.html)
  if (!textInput) return;

  // ---------- Init ----------
  loadHistory();
  renderHistory();

  // ---------- Character counter ----------
  textInput.addEventListener("input", () => {
    charCount.textContent = `${textInput.value.length} / 400`;
    if (textInput.value.trim()) warning.hidden = true;
  });

  // ---------- Example chips ----------
  if (examples) {
    examples.addEventListener("click", (e) => {
      const chip = e.target.closest(".chip");
      if (!chip) return;
      textInput.value = chip.dataset.text;
      charCount.textContent = `${textInput.value.length} / 400`;
      warning.hidden = true;
      textInput.focus();
    });
  }

  // ---------- Clear ----------
  clearBtn.addEventListener("click", () => {
    textInput.value = "";
    charCount.textContent = "0 / 400";
    warning.hidden = true;
    resultCard.hidden = true;
  });

  // ---------- Predict ----------
  predictBtn.addEventListener("click", handlePredict);
  textInput.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") handlePredict();
  });

  async function handlePredict() {
    const text = textInput.value.trim();

    if (!text) {
      warning.hidden = false;
      textInput.focus();
      return;
    }
    warning.hidden = true;
    setLoading(true);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();

      if (!res.ok) {
        warning.textContent = data.error || "Something went wrong. Please try again.";
        warning.hidden = false;
        setLoading(false);
        return;
      }

      renderResult(text, data);
      addToHistory(text, data);
    } catch (err) {
      warning.textContent = "Network error — please check your connection and try again.";
      warning.hidden = false;
    } finally {
      setLoading(false);
    }
  }

  function setLoading(isLoading) {
    predictBtn.disabled = isLoading;
    predictBtn.querySelector(".btn-label").textContent = isLoading ? "Analyzing..." : "Detect Emotion";
    predictBtn.querySelector(".btn-spinner").hidden = !isLoading;
  }

  // ---------- Render result ----------
  function renderResult(text, data) {
    resultCard.hidden = false;
    resultCard.style.animation = "none";
    void resultCard.offsetWidth; // restart animation
    resultCard.style.animation = "";

    resultEmoji.textContent = data.emoji;
    resultEmoji.style.background = `${data.color}22`;
    resultEmotion.textContent = capitalize(data.emotion);
    resultEmotion.style.color = data.color;

    confidenceValue.textContent = `${Math.round(data.confidence)}%`;
    ringFg.style.stroke = data.color;
    const offset = RING_CIRCUMFERENCE - (data.confidence / 100) * RING_CIRCUMFERENCE;
    // Reset then animate
    ringFg.style.transition = "none";
    ringFg.style.strokeDashoffset = RING_CIRCUMFERENCE;
    void ringFg.offsetWidth;
    ringFg.style.transition = "";
    requestAnimationFrame(() => {
      ringFg.style.strokeDashoffset = offset;
    });

    // Probability bars, sorted descending
    const entries = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    probBars.innerHTML = "";
    entries.forEach(([emotion, pct]) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-name">${emotion}</span>
        <span class="prob-track"><span class="prob-fill"></span></span>
        <span class="prob-pct">${pct}%</span>
      `;
      probBars.appendChild(row);
      const fill = row.querySelector(".prob-fill");
      fill.style.background = EMOTION_COLORS[emotion] || "#5B8CFF";
      requestAnimationFrame(() => {
        fill.style.width = `${pct}%`;
      });
    });

    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ---------- History ----------
  function addToHistory(text, data) {
    history.unshift({ text, emotion: data.emotion, emoji: data.emoji, color: data.color, confidence: data.confidence });
    history = history.slice(0, 8);
    saveHistory();
    renderHistory();
  }

  function renderHistory() {
    if (!history.length) {
      historyList.innerHTML = '<li class="history-empty">Your recent predictions will show up here.</li>';
      return;
    }
    historyList.innerHTML = "";
    history.forEach((item) => {
      const li = document.createElement("li");
      li.className = "history-item";
      li.innerHTML = `
        <span class="h-emoji">${item.emoji}</span>
        <span class="h-text">"${escapeHtml(item.text)}"</span>
        <span class="h-tag" style="background:${item.color}22;color:${item.color};">${capitalize(item.emotion)}</span>
      `;
      historyList.appendChild(li);
    });
  }

  function saveHistory() {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch (e) {
      /* localStorage unavailable — history stays in-memory only */
    }
  }

  function loadHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      if (raw) history = JSON.parse(raw);
    } catch (e) {
      history = [];
    }
  }

  if (resetHistoryBtn) {
    resetHistoryBtn.addEventListener("click", () => {
      history = [];
      saveHistory();
      renderHistory();
    });
  }

  // ---------- Utils ----------
  function capitalize(str) {
    return str ? str.charAt(0).toUpperCase() + str.slice(1) : str;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
})();
