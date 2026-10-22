// Voice Calling Agent — browser client.
// Binary uplink: PCM16 @ 16 kHz. Binary downlink: 4-byte turn_id + PCM16 @ out_sample_rate.
// Text frames: JSON control messages.

// Force a diagnostic mode with ?mode=echo or ?mode=vad; default is the real call pipeline.
const MODE = new URLSearchParams(location.search).get("mode") || "call";

const STATE_LABELS = {
  idle: "Ready",
  listening: "Listening",
  user_speaking: "Listening",
  processing: "Thinking",
  speaking: "Speaking",
};

const callBtn = document.getElementById("call-btn");
const startSessionBtn = document.getElementById("start-session-btn");
const liveCallHint = document.getElementById("live-call-hint");
const statusLine = document.getElementById("status-line");
const transcriptEl = document.getElementById("transcript");
const botSelect = document.getElementById("bot-select");
const micIndicator = document.getElementById("mic-indicator");
const micCaption = document.getElementById("mic-caption");
const connectionBadge = document.getElementById("connection-badge");
const liveBadge = document.getElementById("live-badge");
const liveTimerEl = document.getElementById("live-timer");
const muteBtn = document.getElementById("mute-btn");
const endBtn = document.getElementById("end-btn");
let isMuted = false;

const cvCustomerName = document.getElementById("cv-customer-name");
const cvCompanyName = document.getElementById("cv-company-name");
const companyNameField = document.getElementById("company-name-field");
const callVarInputs = [cvCustomerName, cvCompanyName];

let isCollegeVidyaUser = false;

function callVarsFilled() {
  if (isCollegeVidyaUser) {
    return cvCustomerName.value.trim() !== "";
  }
  return cvCustomerName.value.trim() !== "" && cvCompanyName.value.trim() !== "";
}
function refreshCallBtnAvailability() {
  if (live) return;
  const botReady = isAdminUser ? adminBotOptions.length > 0 : myAssignments.length > 0;
  const disabled = !(callVarsFilled() && botReady);
  callBtn.disabled = disabled;
  startSessionBtn.disabled = disabled;
}
callVarInputs.forEach((el) => el.addEventListener("input", refreshCallBtnAvailability));

// Voice-Quality is a custom widget (not a <select>) so styling isn't limited
// to a native <select>'s closed-state options; only the quality tag
// (Core/Enhanced/Premium/Studio) is shown, not the underlying tier/vendor name.
const tierTrigger = document.getElementById("tier-trigger");
const tierTriggerMain = document.getElementById("tier-trigger-main");
const tierList = document.getElementById("tier-list");
const tierSelectWrap = document.getElementById("tier-select-wrap");
const tierBadgePanel = document.getElementById("tier-badge-panel");

// Gender is picked first (a chip dropdown, same widget as Voice Quality/Voice
// Type); Voice Type then lists only that gender's specific voices, showing
// each one's character (e.g. "Warm", "Bright") rather than a raw letter/star-name.
const genderTrigger = document.getElementById("gender-trigger");
const genderTriggerMain = document.getElementById("gender-trigger-main");
const genderList = document.getElementById("gender-list");
const genderSelectWrap = document.getElementById("gender-select-wrap");
const GENDERS = [
  { code: "female", label: "Female" },
  { code: "male", label: "Male" },
];
const voiceTrigger = document.getElementById("voice-trigger");
const voiceTriggerMain = document.getElementById("voice-trigger-main");
const voiceList = document.getElementById("voice-list");
const voiceSelectWrap = document.getElementById("voice-select-wrap");

let currentTiers = [];
let selectedTier = null;
let selectedGender = null;
let selectedVoiceName = null;
let selectedLanguage = null;
let timerInterval = null;
let liveStartTs = null;

let ws = null;
let micStream = null;
let micCtx = null;
let playCtx = null;
let playerNode = null;
let currentTurnId = -1;
let live = false;

function setStatus(text) { statusLine.textContent = text; }

function setLamp(state) {
  const label = STATE_LABELS[state] || state;
  micCaption.textContent = label;
  micIndicator.dataset.state = state;
}

function renderTierTrigger() {
  const tier = currentTiers.find((t) => t.name === selectedTier);
  if (!tier) {
    tierTriggerMain.textContent = "…";
    if (tierBadgePanel) tierBadgePanel.innerHTML = "";
    return;
  }
  tierTriggerMain.textContent = tier.quality;
  if (tierBadgePanel) {
    tierBadgePanel.innerHTML = `<span class="badge-quality">${tier.quality}</span>`;
  }
}

function formatElapsed(ms) {
  const total = Math.floor(ms / 1000);
  const m = String(Math.floor(total / 60)).padStart(2, "0");
  const s = String(total % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function startTimer() {
  liveStartTs = Date.now();
  liveTimerEl.textContent = "00:00";
  liveBadge.classList.remove("hidden");
  connectionBadge.classList.add("live");
  timerInterval = setInterval(() => {
    liveTimerEl.textContent = formatElapsed(Date.now() - liveStartTs);
  }, 1000);
}

function stopTimer() {
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = null;
  liveBadge.classList.add("hidden");
  connectionBadge.classList.remove("live");
}

function setMuted(muted) {
  isMuted = muted;
  if (micStream) micStream.getAudioTracks().forEach((t) => (t.enabled = !muted));
  muteBtn.classList.toggle("active", muted);
  muteBtn.setAttribute("aria-pressed", String(muted));
  muteBtn.title = muted ? "Unmute" : "Mute";
}
muteBtn.addEventListener("click", () => setMuted(!isMuted));
endBtn.addEventListener("click", () => endCall());

function renderTierList() {
  tierList.innerHTML = "";
  for (const tier of currentTiers) {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.dataset.value = tier.name;
    li.innerHTML = `<span class="chip-main">${tier.quality}</span>`;
    li.addEventListener("click", () => {
      selectedTier = tier.name;
      renderTierTrigger();
      syncVoiceControls();
      closeTierList();
    });
    tierList.appendChild(li);
  }
}

function renderGenderTrigger() {
  const g = GENDERS.find((x) => x.code === selectedGender);
  genderTriggerMain.textContent = g ? g.label : "…";
}

function renderGenderList() {
  genderList.innerHTML = "";
  for (const g of GENDERS) {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.classList.toggle("active", g.code === selectedGender);
    li.innerHTML = `<span class="chip-main">${g.label}</span>`;
    li.addEventListener("click", () => {
      selectGender(g.code);
      closeGenderList();
    });
    genderList.appendChild(li);
  }
}

function selectGender(code) {
  selectedGender = code;
  syncVoiceControls();
}

function setGenderControlsDisabled(disabled) {
  genderTrigger.disabled = disabled;
  if (disabled) closeGenderList();
}

function renderVoiceTrigger() {
  const tier = currentTiers.find((t) => t.name === selectedTier);
  const voices = (tier && tier.voices && tier.voices[selectedGender]) || [];
  const voice = voices.find((v) => v.name === selectedVoiceName);
  voiceTriggerMain.textContent = voice ? voice.label : "…";
}

// Only the currently selected gender's voices - gender is chosen first (the
// dropdown above), this just drills one level deeper into it. Each option
// shows only the voice's character word (e.g. "Warm", "Bright") - never the
// raw letter/star-name Google uses internally (e.g. "Kore", "Laomedeia").
function renderVoiceList() {
  voiceList.innerHTML = "";
  const tier = currentTiers.find((t) => t.name === selectedTier);
  const voices = (tier && tier.voices && tier.voices[selectedGender]) || [];
  for (const v of voices) {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.classList.toggle("active", v.name === selectedVoiceName);
    li.innerHTML = `<span class="chip-main">${v.label}</span>`;
    li.addEventListener("click", () => {
      selectedVoiceName = v.name;
      renderVoiceTrigger();
      renderVoiceList();
      closeVoiceList();
    });
    voiceList.appendChild(li);
  }
}

// Keeps gender/voice in sync after the tier or language changes, or after the
// user picks a different gender: keeps the same gender if it's still offered,
// and within it the same specific voice if still offered, else falls back to
// the first option - same pattern as tier/language.
function syncVoiceControls() {
  const tier = currentTiers.find((t) => t.name === selectedTier);
  const availableGenders = (tier && tier.voices) ? Object.keys(tier.voices) : [];
  if (!availableGenders.length) {
    selectedGender = null;
    selectedVoiceName = null;
  } else {
    if (!availableGenders.includes(selectedGender)) {
      selectedGender = availableGenders.includes("female") ? "female" : availableGenders[0];
    }
    const voices = tier.voices[selectedGender] || [];
    if (!voices.some((v) => v.name === selectedVoiceName)) {
      selectedVoiceName = voices[0] ? voices[0].name : null;
    }
  }
  renderGenderTrigger();
  renderGenderList();
  renderVoiceTrigger();
  renderVoiceList();
}
renderGenderTrigger();
renderGenderList();

function openTierList() {
  if (tierTrigger.disabled) return;
  tierList.classList.remove("hidden");
}
function closeTierList() { tierList.classList.add("hidden"); }
function toggleTierList() { tierList.classList.contains("hidden") ? openTierList() : closeTierList(); }

function openVoiceList() {
  if (voiceTrigger.disabled) return;
  voiceList.classList.remove("hidden");
}
function closeVoiceList() { voiceList.classList.add("hidden"); }
function toggleVoiceList() { voiceList.classList.contains("hidden") ? openVoiceList() : closeVoiceList(); }

function openGenderList() {
  if (genderTrigger.disabled) return;
  genderList.classList.remove("hidden");
}
function closeGenderList() { genderList.classList.add("hidden"); }
function toggleGenderList() { genderList.classList.contains("hidden") ? openGenderList() : closeGenderList(); }

tierTrigger.addEventListener("click", toggleTierList);
voiceTrigger.addEventListener("click", toggleVoiceList);
genderTrigger.addEventListener("click", toggleGenderList);
document.addEventListener("click", (e) => {
  if (!tierSelectWrap.contains(e.target)) closeTierList();
  if (!voiceSelectWrap.contains(e.target)) closeVoiceList();
  if (!genderSelectWrap.contains(e.target)) closeGenderList();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeTierList(); closeVoiceList(); closeGenderList(); }
});

function setTierControlsDisabled(disabled) {
  tierTrigger.disabled = disabled;
  if (disabled) closeTierList();
}
function setVoiceControlsDisabled(disabled) {
  voiceTrigger.disabled = disabled;
  if (disabled) closeVoiceList();
}

async function loadTiers(languageCode, assignmentId) {
  if (!languageCode) return;
  try {
    let url = `/tts-options?language=${encodeURIComponent(languageCode)}`;
    if (assignmentId) url += `&assignment_id=${encodeURIComponent(assignmentId)}`;
    const res = await fetch(url);
    const { tiers } = await res.json();
    const previous = selectedTier;
    currentTiers = tiers;
    // keep the same tier selected across a language change if it's still offered
    selectedTier = tiers.some((t) => t.name === previous) ? previous : (tiers[0] ? tiers[0].name : null);
    renderTierTrigger();
    renderTierList();
    syncVoiceControls();
  } catch (err) {
    setStatus("failed to load voice options: " + err.message);
  }
}

// There's no standalone Language control anymore - language is always baked
// into whichever Bot dropdown option is picked. Admins get every bot for
// free-choice testing (loadBots) with one dropdown option per (bot, language)
// pair a bot has a prompt for; everyone else only ever sees their own assigned
// (bot, language) pairs (loadMyBots).
let isAdminUser = false;
let myAssignments = [];
let adminBotOptions = []; // [{value: "botId:langCode", bot_id, language_code}]

async function loadLangLabels() {
  try {
    const langs = await fetchJSON("/languages");
    const labels = {};
    for (const l of langs) labels[l.code] = l.label;
    return labels;
  } catch (err) {
    return {};
  }
}

// Disambiguates same-named bot options that differ only by language (e.g. the
// same bot assigned twice, once per language) with a "(English)"-style suffix.
// Pipeline is a per-call Voice Quality choice now, not a bot property, so it's
// no longer part of this label.
function botOptionLabel(name, langLabel, ambiguousLanguage) {
  return ambiguousLanguage ? `${name} (${langLabel})` : name;
}

function selectAdminBotOption(value) {
  const opt = adminBotOptions.find((o) => o.value === String(value));
  if (!opt) return;
  selectedLanguage = opt.language_code;
  loadTiers(selectedLanguage);
}

async function loadBots() {
  // preserve whatever was already selected across a refresh (e.g. Advanced
  // Settings mutating an unrelated bot) instead of always jumping back to the
  // first option
  const previousValue = botSelect.value;
  try {
    const [bots, langLabels] = await Promise.all([fetchJSON("/api/bots"), loadLangLabels()]);
    adminBotOptions = [];
    botSelect.innerHTML = "";
    for (const b of bots) {
      for (const code of b.languages) {
        const value = `${b.id}:${code}`;
        adminBotOptions.push({ value, bot_id: b.id, language_code: code });
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = botOptionLabel(b.name, langLabels[code] || code, b.languages.length > 1);
        botSelect.appendChild(opt);
      }
    }
    const stillExists = adminBotOptions.some((o) => o.value === previousValue);
    if (stillExists) {
      botSelect.value = previousValue;
      selectAdminBotOption(previousValue);
    } else if (adminBotOptions[0]) {
      selectAdminBotOption(adminBotOptions[0].value);
    }
    if (!adminBotOptions.length) setStatus("no bots exist yet - create one in Advanced Settings");
    refreshCallBtnAvailability();
  } catch (err) {
    setStatus("failed to load bots: " + err.message);
  }
}

function selectAssignment(assignmentId) {
  const a = myAssignments.find((x) => String(x.id) === String(assignmentId));
  if (!a) return;
  selectedLanguage = a.language_code;
  loadTiers(selectedLanguage, a.id);
}

async function loadMyBots() {
  try {
    myAssignments = await fetchJSON("/api/my-bots");
  } catch (err) {
    myAssignments = [];
    setStatus("failed to load your bots: " + err.message);
  }
  botSelect.innerHTML = "";
  for (const a of myAssignments) {
    const opt = document.createElement("option");
    opt.value = a.id;
    // disambiguate only if the same bot appears twice (assigned in two languages)
    const ambiguous = myAssignments.some((x) => x.bot_id === a.bot_id && x.id !== a.id);
    opt.textContent = botOptionLabel(a.bot_name, a.language_label, ambiguous);
    botSelect.appendChild(opt);
  }
  if (myAssignments[0]) selectAssignment(myAssignments[0].id);
  if (!myAssignments.length) setStatus("no bots assigned yet - contact your admin");
  refreshCallBtnAvailability();
}

botSelect.addEventListener("change", () => {
  if (isAdminUser) selectAdminBotOption(botSelect.value);
  else selectAssignment(botSelect.value);
});

window.currentUserPromise.then((user) => {
  if (!user) return;
  isAdminUser = user.role === "admin";
  if (user.username && user.username.toLowerCase().includes("collegevidya")) {
    isCollegeVidyaUser = true;
    if (companyNameField) companyNameField.style.display = "none";
    if (cvCompanyName) cvCompanyName.removeAttribute("required");
  }
  if (isAdminUser) {
    loadBots();
  } else {
    loadMyBots();
  }
  refreshCallBtnAvailability();
});

function clearTranscript() {
  transcriptEl.innerHTML = `
    <div class="transcript-empty">
      <div class="transcript-empty-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path>
        </svg>
      </div>
      <p class="hint">Your conversation will appear here<br>once you start the session.</p>
    </div>`;
}

function addMessage(role, text) {
  const empty = transcriptEl.querySelector(".transcript-empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = "msg " + role;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "user" ? "You" : "Agent";
  div.appendChild(who);
  div.appendChild(document.createTextNode(text));
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return div;
}

// Agent text arrives sentence-by-sentence; append to the current agent bubble.
let agentBubble = null;
function appendAgentText(turnId, text) {
  if (!agentBubble || agentBubble.dataset.turnId !== String(turnId)) {
    agentBubble = addMessage("agent", text);
    agentBubble.dataset.turnId = String(turnId);
  } else {
    agentBubble.appendChild(document.createTextNode(" " + text));
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }
}

async function setupAudio(outSampleRate) {
  // Playback context at the server's output rate; the browser handles hardware conversion.
  playCtx = new AudioContext({ sampleRate: outSampleRate });
  await playCtx.audioWorklet.addModule("/static/player-worklet.js");
  playerNode = new AudioWorkletNode(playCtx, "pcm-player", {
    numberOfInputs: 0,
    outputChannelCount: [1],
  });
  playerNode.port.onmessage = (e) => {
    // the tab fell more than 12s behind real-time (e.g. backgrounded/throttled)
    // and the player had to skip ahead - surfaced here since it would
    // otherwise be a silent, undiagnosable audio glitch
    if (e.data.event === "overflow") console.warn("audio playback buffer overflowed - skipped ahead to catch up");
  };
  playerNode.connect(playCtx.destination);

  // Mic context at the hardware rate; the worklet downsamples to 16 kHz.
  micCtx = new AudioContext();
  await micCtx.audioWorklet.addModule("/static/mic-worklet.js");
  const source = micCtx.createMediaStreamSource(micStream);
  const micNode = new AudioWorkletNode(micCtx, "mic-capture", {
    numberOfOutputs: 1,
    processorOptions: { targetRate: 16000, frameSamples: 512 },
  });
  micNode.port.onmessage = (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
  };
  source.connect(micNode);
  // Keep the node in the processing graph without hearing yourself directly.
  const mute = micCtx.createGain();
  mute.gain.value = 0;
  micNode.connect(mute);
  mute.connect(micCtx.destination);
}

function handleControl(msg) {
  switch (msg.type) {
    case "ready":
      setupAudio(msg.out_sample_rate)
        .then(() => setStatus(msg.mode === "echo"
          ? "echo mode - you should hear yourself"
          : "connected"))
        .catch((err) => { setStatus("audio setup failed: " + err.message); endCall(); });
      break;
    case "state":
      setLamp(msg.state);
      break;
    case "turn_start":
      currentTurnId = msg.turn_id;
      break;
    case "transcript":
      addMessage("user", msg.text);
      break;
    case "agent_text":
      appendAgentText(msg.turn_id, msg.text);
      break;
    case "barge_in":
      if (playerNode) playerNode.port.postMessage({ cmd: "flush" });
      currentTurnId = -1; // drop any in-flight frames until the next turn_start
      if (agentBubble && !agentBubble.classList.contains("interrupted")) {
        agentBubble.classList.add("interrupted");
        const tag = document.createElement("span");
        tag.className = "badge interrupted-tag";
        tag.textContent = "Interrupted";
        agentBubble.appendChild(tag);
      }
      break;
    case "turn_end":
      break;
    case "vad_utterance":
      setStatus(`utterance captured: ${msg.duration_s}s${msg.forced ? " (max length)" : ""}`);
      break;
    case "error":
      setStatus("error: " + msg.message);
      break;
  }
}

async function startCall() {
  // defensive - the button is disabled until both are true anyway
  if (!callVarsFilled() || (!isAdminUser && !myAssignments.length)) return;
  callBtn.disabled = true;
  startSessionBtn.disabled = true;
  botSelect.disabled = true;
  setTierControlsDisabled(true);
  setGenderControlsDisabled(true);
  setVoiceControlsDisabled(true);
  callVarInputs.forEach((el) => (el.disabled = true));
  clearTranscript();
  try {
    setStatus("requesting microphone...");
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,   // required for barge-in (keeps agent audio out of the mic)
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
  } catch (err) {
    setStatus("microphone denied: " + err.message);
    botSelect.disabled = false;
    setTierControlsDisabled(false);
    setGenderControlsDisabled(false);
    setVoiceControlsDisabled(false);
    callVarInputs.forEach((el) => (el.disabled = false));
    refreshCallBtnAvailability();
    return;
  }

  setStatus("connecting...");
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    ws.send(JSON.stringify({
      type: "start",
      mode: MODE,
      // admins free-pick a bot+language combo (encoded in the Bot dropdown's
      // value); everyone else picks one of their own assignments, which the
      // server resolves to a bot+language itself
      bot_id: isAdminUser ? botSelect.value.split(":")[0] : null,
      assignment_id: isAdminUser ? null : botSelect.value,
      language: selectedLanguage,
      gender: selectedGender,
      tts_tier: selectedTier,
      tts_voice: selectedVoiceName,
      customer_name: cvCustomerName.value.trim(),
      company_name: isCollegeVidyaUser ? (cvCompanyName.value.trim() || "College Vidya") : cvCompanyName.value.trim(),
    }));
    live = true;
    callBtn.disabled = true;
    startSessionBtn.disabled = true;
    muteBtn.disabled = false;
    endBtn.disabled = false;
    liveCallHint.classList.add("hidden");
    startTimer();
  };

  ws.onmessage = (e) => {
    if (typeof e.data === "string") {
      handleControl(JSON.parse(e.data));
      return;
    }
    // binary: 4-byte LE turn_id + PCM16
    const view = new DataView(e.data);
    const turnId = view.getUint32(0, true);
    if (turnId !== currentTurnId || !playerNode) return;
    const pcm = e.data.slice(4);
    playerNode.port.postMessage({ cmd: "push", pcm }, [pcm]);
  };

  ws.onclose = () => { if (live) endCall(); };
  ws.onerror = () => setStatus("connection error");
}

function endCall() {
  live = false;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
    ws.close();
  }
  ws = null;
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  if (micCtx) { micCtx.close(); micCtx = null; }
  if (playCtx) { playCtx.close(); playCtx = null; playerNode = null; }
  currentTurnId = -1;
  agentBubble = null;
  setMuted(false);
  muteBtn.disabled = true;
  endBtn.disabled = true;
  liveCallHint.classList.remove("hidden");
  botSelect.disabled = false;
  setTierControlsDisabled(false);
  setGenderControlsDisabled(false);
  setVoiceControlsDisabled(false);
  callVarInputs.forEach((el) => (el.disabled = false));
  refreshCallBtnAvailability();
  stopTimer();
  setLamp("idle");
  setStatus("call ended");
}

callBtn.addEventListener("click", () => { if (!live) startCall(); });
startSessionBtn.addEventListener("click", () => { if (!live) startCall(); });
