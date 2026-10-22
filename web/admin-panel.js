// Advanced Settings: a dedicated page (Users / Agent Prompt tabs), admin-only.
// Renders into #admin-panel-root on settings.html - navigated to from the main
// call page's topbar, not opened as a modal over it.

function adminEscapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatUserDate(text) {
  if (!text) return "—";
  const d = new Date(text.replace(" ", "T") + "Z"); // sqlite CURRENT_TIMESTAMP-style string
  return isNaN(d.getTime()) ? text : d.toLocaleString();
}

(async function initAdminPanel() {
  const user = await window.currentUserPromise;
  if (!user) return; // currentUserPromise already redirects to /login in this case
  if (user.role !== "admin") {
    location.href = "/"; // this whole page is admin-only
    return;
  }

  const root = document.getElementById("admin-panel-root");
  if (!root) return;

  const selfId = user.id;

  // ---- main panel content -----------------------------------------------------

  const overlay = root; // kept as `overlay` - the rest of this file scopes queries to it
  overlay.innerHTML = `
      <div class="tabs">
        <button class="tab-btn active" data-tab="users">Users</button>
        <button class="tab-btn" data-tab="bots">Bots</button>
        <button class="tab-btn" data-tab="calllogs">Call Logs</button>
      </div>
      <div class="modal-body">
        <div class="tab-panel active" data-panel="users">
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 0.9rem;">
            <p style="margin:0; color: var(--text-muted); font-size: 0.85rem;">Accounts that can log in to this app.</p>
            <button id="add-user-btn" class="btn btn-primary">+ Add user</button>
          </div>
          <div class="table-wrap">
            <table class="admin-table users-table">
              <colgroup>
                <col class="col-username"><col class="col-role"><col class="col-bots">
                <col class="col-created"><col class="col-actions">
              </colgroup>
              <thead>
                <tr><th>Username</th><th>Role</th><th>Bots</th><th>Created</th><th>Actions</th></tr>
              </thead>
              <tbody id="admin-users-body"></tbody>
            </table>
          </div>
        </div>
        <div class="tab-panel" data-panel="bots">
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 0.9rem;">
            <p style="margin:0; color: var(--text-muted); font-size: 0.85rem;">
              Named prompt sets - each bot can have a different prompt per language.
              Assign any number of (bot, language) pairs to a user from the Users tab;
              that user picks between only their own assigned bots from the Bot dropdown
              and never sees the language control (it's fixed by the assignment). Admins
              get a free-choice Bot + Language selector instead, for testing.
            </p>
            <button id="add-bot-btn" class="btn btn-primary">+ New bot</button>
          </div>
          <div class="table-wrap">
            <table class="admin-table bots-table">
              <colgroup>
                <col class="col-name"><col class="col-prompts"><col class="col-created"><col class="col-actions">
              </colgroup>
              <thead>
                <tr><th>Name</th><th>Prompts set</th><th>Created</th><th>Actions</th></tr>
              </thead>
              <tbody id="admin-bots-body"></tbody>
            </table>
          </div>

          <div id="bot-editor" style="display:none; margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px solid var(--border);">
            <div class="field">
              <label for="bot-editor-name">Bot name</label>
              <div style="display:flex; gap:0.5rem;">
                <input id="bot-editor-name" style="flex:1;">
                <button id="bot-editor-save-name" class="btn">Rename</button>
              </div>
            </div>
            <div class="lang-pills" id="bot-editor-langs"></div>
            <p style="margin: 0.6rem 0 1rem; color: var(--text-muted); font-size: 0.85rem;">
              This is the persona/behavior instructions given to the agent for the
              language selected above - who it is, how it should talk, what it should
              do. The app automatically appends a language-lock rule ("reply only in
              Hindi", etc.), the read-aloud formatting rule, and which voice gender was
              picked for the call - you don't need to write those yourself. To use a
              different name per gender, write it into this prompt explicitly, e.g. "If
              the voice is female, your name is Asha. If it is male, your name is
              Rohan." Use <code>{{customer_name}}</code>/<code>{{company_name}}</code>
              to reference the per-call "Call details" fields.
            </p>
            <textarea id="bot-editor-prompt" spellcheck="false" style="width:100%; min-height: 260px; resize: vertical; border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem; font-size: 0.92rem; line-height: 1.5; font-family: inherit;"></textarea>
            <p id="bot-editor-error" class="form-error" style="display: none;"></p>
            <div style="display:flex; align-items:center; gap: 0.9rem; margin-top: 0.9rem;">
              <button id="bot-editor-save-prompt" class="btn btn-primary">Save prompt</button>
              <span id="bot-editor-status" style="display: none; font-size: 0.85rem; color: var(--green);">Saved.</span>
            </div>
          </div>
        </div>
        <div class="tab-panel" data-panel="calllogs">
          <p style="margin:0 0 0.9rem; color: var(--text-muted); font-size: 0.85rem;">
            Every call placed through this app - who called, what tier/language, and the
            full turn-by-turn transcript of what the caller and the agent said.
          </p>
          <div class="table-wrap">
            <table class="admin-table calllogs-table">
              <colgroup>
                <col class="col-username"><col class="col-when"><col class="col-lang">
                <col class="col-tier"><col class="col-duration"><col class="col-cost"><col class="col-actions">
              </colgroup>
              <thead>
                <tr><th>User</th><th>When</th><th>Language</th><th>Voice Quality</th><th>Duration</th><th>Cost</th><th>Recording</th><th>Actions</th></tr>
              </thead>
              <tbody id="admin-calllogs-body"></tbody>
            </table>
          </div>
          <p id="calllogs-empty" class="empty-state" style="display: none;">No calls logged yet.</p>
        </div>
      </div>
  `;

  const addUserOverlay = document.createElement("div");
  addUserOverlay.className = "modal-overlay hidden";
  addUserOverlay.innerHTML = `
    <div class="modal">
      <div class="modal-header">
        <h2>Add user</h2>
        <button class="modal-close" id="admin-add-user-close">&times;</button>
      </div>
      <div class="modal-body">
        <form id="admin-add-user-form">
          <div class="field">
            <label for="admin-new-username">Username</label>
            <input id="admin-new-username" required>
          </div>
          <div class="field">
            <label for="admin-new-password">Temporary password</label>
            <input id="admin-new-password" type="password" required>
          </div>
          <div class="field">
            <label for="admin-new-role">Role</label>
            <select id="admin-new-role">
              <option value="user">User</option>
              <option value="admin">Admin</option>
            </select>
          </div>
          <p id="admin-add-user-error" class="form-error" style="display: none;"></p>
          <button type="submit" class="btn btn-primary" style="width: 100%;">Create user</button>
        </form>
      </div>
    </div>
  `;
  document.body.appendChild(addUserOverlay);

  const manageBotsOverlay = document.createElement("div");
  manageBotsOverlay.className = "modal-overlay hidden";
  manageBotsOverlay.innerHTML = `
    <div class="modal">
      <div class="modal-header">
        <h2 id="manage-bots-title">Bots for user</h2>
        <button class="modal-close" id="admin-manage-bots-close">&times;</button>
      </div>
      <div class="modal-body">
        <form id="manage-bots-add-form">
          <div class="assign-bot-form">
            <select id="manage-bots-bot-select"></select>
            <select id="manage-bots-lang-select"></select>
            <button type="submit" class="btn btn-primary">Add</button>
          </div>
          <p style="margin: 0.6rem 0 0.3rem; color: var(--text-muted); font-size: 0.8rem;">
            Voice qualities to allow (pick at least one):
          </p>
          <div id="manage-bots-tier-checks"></div>
        </form>
        <p id="manage-bots-error" class="form-error" style="display: none; margin-top: 0.5rem;"></p>
        <div id="manage-bots-list" class="assigned-bots-list" style="margin-top: 1rem;"></div>
        <p id="manage-bots-empty" class="empty-state" style="display: none; padding: 0.5rem 0 0;">No bots assigned yet.</p>
      </div>
    </div>
  `;
  document.body.appendChild(manageBotsOverlay);

  const addBotOverlay = document.createElement("div");
  addBotOverlay.className = "modal-overlay hidden";
  addBotOverlay.innerHTML = `
    <div class="modal">
      <div class="modal-header">
        <h2>New bot</h2>
        <button class="modal-close" id="admin-add-bot-close">&times;</button>
      </div>
      <div class="modal-body">
        <form id="admin-add-bot-form">
          <div class="field">
            <label for="admin-new-bot-name">Name</label>
            <input id="admin-new-bot-name" required>
          </div>
          <p id="admin-add-bot-error" class="form-error" style="display: none;"></p>
          <button type="submit" class="btn btn-primary" style="width: 100%;">Create bot</button>
        </form>
      </div>
    </div>
  `;
  document.body.appendChild(addBotOverlay);

  const callLogDetailOverlay = document.createElement("div");
  callLogDetailOverlay.className = "modal-overlay hidden";
  callLogDetailOverlay.innerHTML = `
    <div class="modal" style="max-width: 720px;">
      <div class="modal-header">
        <h2 id="calllog-detail-title">Call transcript</h2>
        <button class="modal-close" id="calllog-detail-close">&times;</button>
      </div>
      <div class="modal-body">
        <div id="calllog-detail-meta" style="margin-bottom: 1rem; font-size: 0.85rem; color: var(--text-muted);"></div>
        <div id="calllog-detail-transcript" style="display:flex; flex-direction:column; gap:0.6rem; max-height: 55vh; overflow-y:auto;"></div>
      </div>
    </div>
  `;
  document.body.appendChild(callLogDetailOverlay);

  // ---- tabs ------------------------------------------------------------------

  overlay.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      overlay.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      overlay.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      overlay.querySelector(`.tab-panel[data-panel="${btn.dataset.tab}"]`).classList.add("active");
    });
  });

  // ---- users tab -----------------------------------------------------------------

  const usersBody = document.getElementById("admin-users-body");
  const addUserBtn = document.getElementById("add-user-btn");
  const addUserClose = document.getElementById("admin-add-user-close");
  const addUserForm = document.getElementById("admin-add-user-form");
  const addUserError = document.getElementById("admin-add-user-error");

  // Bots don't apply to admin accounts at all (they get the free-choice topbar
  // Bot + Language selector instead) - the "Bots" column/management modal below
  // only matters for role="user".
  function renderUserRow(u) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${adminEscapeHtml(u.username)}</td>
      <td><span class="badge ${u.role === "admin" ? "admin" : ""}">${u.role}</span></td>
      <td class="user-bot-cell"></td>
      <td>${formatUserDate(u.created_at)}</td>
      <td><div class="row-actions"></div></td>
    `;
    const botCell = tr.querySelector(".user-bot-cell");
    if (u.role === "admin") {
      botCell.textContent = "—";
      botCell.style.color = "var(--text-muted)";
    } else if (!u.bots.length) {
      botCell.textContent = "none assigned";
      botCell.style.color = "var(--text-muted)";
    } else {
      const badgesWrap = document.createElement("div");
      badgesWrap.className = "assigned-bots-badges";
      for (const a of u.bots) {
        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = `${a.bot_name} (${a.language_code.toUpperCase()})`;
        badgesWrap.appendChild(badge);
      }
      botCell.appendChild(badgesWrap);
    }
    const actions = tr.querySelector(".row-actions");
    if (u.role !== "admin") {
      const manageBtn = document.createElement("button");
      manageBtn.className = "btn";
      manageBtn.textContent = "Bots";
      manageBtn.addEventListener("click", () => openManageBotsModal(u));
      actions.appendChild(manageBtn);
    }
    const deleteBtn = document.createElement("button");
    deleteBtn.className = "btn btn-danger";
    deleteBtn.textContent = "Delete";
    if (u.id === selfId) {
      deleteBtn.disabled = true;
      deleteBtn.title = "You can't delete your own account while logged in as it";
    } else {
      deleteBtn.addEventListener("click", () => deleteUser(u));
    }
    actions.appendChild(deleteBtn);
    return tr;
  }

  async function loadUsers() {
    usersBody.innerHTML = "";
    try {
      const users = await fetchJSON("/api/users");
      for (const u of users) usersBody.appendChild(renderUserRow(u));
    } catch (err) {
      usersBody.innerHTML = `<tr><td colspan="5" class="form-error">failed to load users: ${err.message}</td></tr>`;
    }
  }

  async function deleteUser(u) {
    if (!confirm(`Delete user "${u.username}"? They'll be logged out immediately.`)) return;
    try {
      await fetchJSON(`/api/users/${u.id}`, { method: "DELETE" });
      await loadUsers();
    } catch (err) {
      alert("failed to delete: " + err.message);
    }
  }

  addUserBtn.addEventListener("click", () => {
    addUserError.style.display = "none";
    addUserForm.reset();
    addUserOverlay.classList.remove("hidden");
  });
  addUserClose.addEventListener("click", () => addUserOverlay.classList.add("hidden"));
  addUserOverlay.addEventListener("click", (e) => { if (e.target === addUserOverlay) addUserOverlay.classList.add("hidden"); });

  addUserForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    addUserError.style.display = "none";
    const role = document.getElementById("admin-new-role").value;
    try {
      await fetchJSON("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: document.getElementById("admin-new-username").value,
          password: document.getElementById("admin-new-password").value,
          role,
        }),
      });
      addUserOverlay.classList.add("hidden");
      await loadUsers();
    } catch (err) {
      addUserError.textContent = err.message;
      addUserError.style.display = "block";
    }
  });

  // ---- manage bots modal (per-user bot+language assignments) ------------------

  const manageBotsClose = document.getElementById("admin-manage-bots-close");
  const manageBotsTitle = document.getElementById("manage-bots-title");
  const manageBotsList = document.getElementById("manage-bots-list");
  const manageBotsEmpty = document.getElementById("manage-bots-empty");
  const manageBotsAddForm = document.getElementById("manage-bots-add-form");
  const manageBotsBotSelect = document.getElementById("manage-bots-bot-select");
  const manageBotsLangSelect = document.getElementById("manage-bots-lang-select");
  const manageBotsTierChecks = document.getElementById("manage-bots-tier-checks");
  const manageBotsError = document.getElementById("manage-bots-error");

  let manageBotsUser = null;
  // [{quality, tierNames}, ...] - the Voice Quality universe grouped by
  // caller-facing label (each label maps to one underlying tier today, but
  // grouping keeps this working if a label ever covers more than one again).
  let tierGroups = [];

  async function loadTierOptions() {
    if (tierGroups.length || enabledLanguages.length === 0) return;
    try {
      const { tiers } = await fetchJSON(`/tts-options?language=${encodeURIComponent(enabledLanguages[0].code)}`);
      tierGroups = [];
      for (const t of tiers) {
        let g = tierGroups.find((g) => g.quality === t.quality);
        if (!g) { g = { quality: t.quality, tierNames: [] }; tierGroups.push(g); }
        g.tierNames.push(t.name);
      }
    } catch (err) {
      tierGroups = [];
    }
  }

  // Always-visible toggle pills (same look as the bot editor's language
  // pills), not a dropdown - every Voice Quality tag sits right below the
  // bot/language/Add row so there's nothing to open before picking one.
  // `checkedNames` narrows which start selected when editing an existing
  // assignment (null there means "currently unrestricted", so every tag
  // starts selected to reflect that true saved state). A fresh "Add" form
  // has no saved state to reflect, so it passes startEmpty:true instead -
  // nothing is picked yet, forcing the admin to actively choose what to
  // allow rather than silently defaulting to "everything".
  function buildTierPills(checkedNames, { startEmpty = false } = {}) {
    const wrap = document.createElement("div");
    wrap.className = "lang-pills";
    for (const g of tierGroups) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "lang-pill";
      btn.textContent = g.quality;
      btn.dataset.quality = g.quality;
      const selected = checkedNames ? g.tierNames.some((n) => checkedNames.includes(n)) : !startEmpty;
      btn.classList.toggle("active", selected);
      btn.addEventListener("click", () => btn.classList.toggle("active"));
      wrap.appendChild(btn);
    }
    wrap.collectAllowedTiers = () => {
      const pills = [...wrap.querySelectorAll(".lang-pill")];
      const selectedQualities = pills.filter((b) => b.classList.contains("active")).map((b) => b.dataset.quality);
      // null (not an empty array) means "every tier allowed" to the API - so a
      // tier added later is included automatically instead of silently excluded
      if (selectedQualities.length === pills.length) return null;
      return tierGroups.filter((g) => selectedQualities.includes(g.quality)).flatMap((g) => g.tierNames);
    };
    return wrap;
  }

  function renderTierChecks(container, checkedNames, opts) {
    container.innerHTML = "";
    container.appendChild(buildTierPills(checkedNames, opts));
  }

  function collectCheckedTiers(container) {
    return container.firstElementChild ? container.firstElementChild.collectAllowedTiers() : null;
  }

  function populateSelectOptions(select, items, valueKey, labelKey) {
    select.innerHTML = "";
    for (const item of items) {
      const opt = document.createElement("option");
      opt.value = item[valueKey];
      opt.textContent = item[labelKey];
      select.appendChild(opt);
    }
  }

  async function loadManageBotsList() {
    manageBotsList.innerHTML = "";
    let assignments;
    try {
      assignments = await fetchJSON(`/api/users/${manageBotsUser.id}/bots`);
    } catch (err) {
      manageBotsList.innerHTML = `<p class="form-error">failed to load: ${err.message}</p>`;
      return;
    }
    manageBotsEmpty.style.display = assignments.length ? "none" : "block";
    for (const a of assignments) {
      const row = document.createElement("div");
      row.className = "assigned-bot-row";
      row.style.flexWrap = "wrap";
      // Unrestricted isn't just another value in this list - it's the one state
      // an admin most needs to notice, since it means the assignment silently
      // allows every Voice Quality tier (including any added in the future).
      // Flag it in the warning color instead of the same muted tone as a real,
      // deliberately-picked tier summary.
      const tierSummary = a.allowed_tiers
        ? tierGroups.filter((g) => g.tierNames.some((n) => a.allowed_tiers.includes(n))).map((g) => g.quality).join(", ")
        : "⚠ Unrestricted - all voice qualities allowed";
      const tierSummaryColor = a.allowed_tiers ? "var(--text-muted)" : "var(--red)";
      row.innerHTML = `<span>${adminEscapeHtml(a.bot_name)} <span class="badge">${a.language_code.toUpperCase()}</span>
        <span style="color: ${tierSummaryColor}; font-size: 0.8rem;"> - ${adminEscapeHtml(tierSummary)}</span></span>`;
      const editWrap = document.createElement("div");
      editWrap.style.display = "flex";
      editWrap.style.gap = "0.5rem";
      const editTiersBtn = document.createElement("button");
      editTiersBtn.type = "button";
      editTiersBtn.className = "btn";
      editTiersBtn.textContent = "Voice qualities";
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "btn btn-danger";
      removeBtn.textContent = "Remove";
      removeBtn.addEventListener("click", async () => {
        try {
          await fetchJSON(`/api/users/${manageBotsUser.id}/bots/${a.bot_id}/${a.language_code}`, { method: "DELETE" });
          await loadManageBotsList();
          await loadUsers();
        } catch (err) {
          alert("failed to remove: " + err.message);
        }
      });
      editWrap.appendChild(editTiersBtn);
      editWrap.appendChild(removeBtn);
      row.appendChild(editWrap);

      const editPanel = document.createElement("div");
      editPanel.style.display = "none";
      editPanel.style.width = "100%";
      editPanel.style.alignItems = "center";
      editPanel.style.gap = "0.6rem";
      editPanel.style.flexWrap = "wrap";
      editPanel.style.margin = "0.6rem 0 0";
      const checksDiv = document.createElement("div");
      checksDiv.className = "tier-checks";
      renderTierChecks(checksDiv, a.allowed_tiers);
      const saveBtn = document.createElement("button");
      saveBtn.type = "button";
      saveBtn.className = "lang-pill active";
      saveBtn.style.marginLeft = "auto"; // sits apart from the tier pills so it doesn't read as one more toggle
      saveBtn.textContent = "✓ Save";
      saveBtn.addEventListener("click", async () => {
        try {
          await fetchJSON(`/api/users/${manageBotsUser.id}/bots/${a.id}/tiers`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ allowed_tiers: collectCheckedTiers(checksDiv) }),
          });
          await loadManageBotsList();
        } catch (err) {
          alert("failed to save voice qualities: " + err.message);
        }
      });
      editPanel.appendChild(checksDiv);
      editPanel.appendChild(saveBtn);
      editTiersBtn.addEventListener("click", () => {
        editPanel.style.display = editPanel.style.display === "none" ? "flex" : "none";
      });
      row.appendChild(editPanel);

      manageBotsList.appendChild(row);
    }
  }

  async function openManageBotsModal(u) {
    manageBotsUser = u;
    manageBotsTitle.textContent = `Bots for ${u.username}`;
    manageBotsError.style.display = "none";
    if (enabledLanguages.length === 0) await loadEnabledLanguages();
    await loadTierOptions();
    populateSelectOptions(manageBotsBotSelect, allBots, "id", "name");
    populateSelectOptions(manageBotsLangSelect, enabledLanguages, "code", "label");
    renderTierChecks(manageBotsTierChecks, null, { startEmpty: true });
    await loadManageBotsList();
    manageBotsOverlay.classList.remove("hidden");
  }

  manageBotsClose.addEventListener("click", () => manageBotsOverlay.classList.add("hidden"));
  manageBotsOverlay.addEventListener("click", (e) => { if (e.target === manageBotsOverlay) manageBotsOverlay.classList.add("hidden"); });

  manageBotsAddForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    manageBotsError.style.display = "none";
    try {
      await fetchJSON(`/api/users/${manageBotsUser.id}/bots`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bot_id: parseInt(manageBotsBotSelect.value, 10),
          language_code: manageBotsLangSelect.value,
          allowed_tiers: collectCheckedTiers(manageBotsTierChecks),
        }),
      });
      renderTierChecks(manageBotsTierChecks, null, { startEmpty: true });
      await loadManageBotsList();
      await loadUsers();
    } catch (err) {
      manageBotsError.textContent = err.message;
      manageBotsError.style.display = "block";
    }
  });

  // ---- bots tab ---------------------------------------------------------------

  const botsBody = document.getElementById("admin-bots-body");
  const addBotBtn = document.getElementById("add-bot-btn");
  const addBotClose = document.getElementById("admin-add-bot-close");
  const addBotForm = document.getElementById("admin-add-bot-form");
  const addBotError = document.getElementById("admin-add-bot-error");

  const botEditor = document.getElementById("bot-editor");
  const botEditorNameInput = document.getElementById("bot-editor-name");
  const botEditorSaveNameBtn = document.getElementById("bot-editor-save-name");
  const botEditorLangs = document.getElementById("bot-editor-langs");
  const botEditorPrompt = document.getElementById("bot-editor-prompt");
  const botEditorSavePromptBtn = document.getElementById("bot-editor-save-prompt");
  const botEditorStatus = document.getElementById("bot-editor-status");
  const botEditorError = document.getElementById("bot-editor-error");

  let enabledLanguages = [];
  let allBots = [];
  let currentBotDetail = null;
  let currentEditorLang = null;

  async function loadEnabledLanguages() {
    try {
      enabledLanguages = await fetchJSON("/languages");
    } catch (err) {
      enabledLanguages = [];
    }
  }

  function renderBotRow(b) {
    const tr = document.createElement("tr");
    const promptsLabel = b.languages.length ? b.languages.join(", ").toUpperCase() : "none yet";
    tr.innerHTML = `
      <td>${adminEscapeHtml(b.name)}</td>
      <td>${promptsLabel}</td>
      <td>${formatUserDate(b.created_at)}</td>
      <td><div class="row-actions"></div></td>
    `;
    const actions = tr.querySelector(".row-actions");
    const editBtn = document.createElement("button");
    editBtn.className = "btn";
    editBtn.textContent = "Edit";
    editBtn.addEventListener("click", () => openBotEditor(b.id));
    const deleteBtn = document.createElement("button");
    deleteBtn.className = "btn btn-danger";
    deleteBtn.textContent = "Delete";
    deleteBtn.addEventListener("click", () => deleteBot(b));
    actions.appendChild(editBtn);
    actions.appendChild(deleteBtn);
    return tr;
  }

  async function loadBotsList() {
    botsBody.innerHTML = "";
    try {
      allBots = await fetchJSON("/api/bots");
      for (const b of allBots) botsBody.appendChild(renderBotRow(b));
    } catch (err) {
      botsBody.innerHTML = `<tr><td colspan="4" class="form-error">failed to load bots: ${err.message}</td></tr>`;
    }
    // app.js's topbar Bot dropdown only loads bots once on page load - every
    // create/rename/delete change in this modal calls loadBotsList() right
    // after, so refreshing it here keeps the dropdown in sync without a
    // hard reload (this modal is admin-only, so the admin free-choice loader,
    // not loadMyBots(), is always the right one to call).
    if (window.loadBots) window.loadBots();
  }

  function updatePillActive() {
    botEditorLangs.querySelectorAll(".lang-pill").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.code === currentEditorLang);
    });
  }

  function selectEditorLang(code) {
    currentEditorLang = code;
    updatePillActive();
    botEditorPrompt.value = (currentBotDetail.prompts && currentBotDetail.prompts[code]) || "";
    botEditorStatus.style.display = "none";
    botEditorError.style.display = "none";
  }

  function renderLangPills() {
    botEditorLangs.innerHTML = "";
    for (const lang of enabledLanguages) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "lang-pill";
      btn.textContent = lang.label;
      btn.dataset.code = lang.code;
      btn.addEventListener("click", () => selectEditorLang(lang.code));
      botEditorLangs.appendChild(btn);
    }
  }

  async function openBotEditor(botId) {
    botEditorError.style.display = "none";
    botEditorStatus.style.display = "none";
    if (enabledLanguages.length === 0) await loadEnabledLanguages();
    try {
      currentBotDetail = await fetchJSON(`/api/bots/${botId}`);
    } catch (err) {
      alert("failed to load bot: " + err.message);
      return;
    }
    botEditorNameInput.value = currentBotDetail.name;
    renderLangPills();
    selectEditorLang(enabledLanguages[0] ? enabledLanguages[0].code : null);
    botEditor.style.display = "block";
    botEditor.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  async function deleteBot(b) {
    if (!confirm(`Delete bot "${b.name}"? This removes all of its prompts.`)) return;
    try {
      await fetchJSON(`/api/bots/${b.id}`, { method: "DELETE" });
      if (currentBotDetail && currentBotDetail.id === b.id) {
        botEditor.style.display = "none";
        currentBotDetail = null;
      }
      await loadBotsList();
    } catch (err) {
      alert("failed to delete: " + err.message);
    }
  }

  botEditorSaveNameBtn.addEventListener("click", async () => {
    botEditorError.style.display = "none";
    if (!currentBotDetail) return;
    const name = botEditorNameInput.value.trim();
    if (!name) return;
    botEditorSaveNameBtn.disabled = true;
    try {
      await fetchJSON(`/api/bots/${currentBotDetail.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      currentBotDetail.name = name;
      await loadBotsList();
    } catch (err) {
      botEditorError.textContent = err.message;
      botEditorError.style.display = "block";
    } finally {
      botEditorSaveNameBtn.disabled = false;
    }
  });

  botEditorSavePromptBtn.addEventListener("click", async () => {
    botEditorError.style.display = "none";
    botEditorStatus.style.display = "none";
    if (!currentBotDetail || !currentEditorLang) return;
    botEditorSavePromptBtn.disabled = true;
    try {
      await fetchJSON(`/api/bots/${currentBotDetail.id}/prompt`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: currentEditorLang, prompt: botEditorPrompt.value }),
      });
      currentBotDetail.prompts[currentEditorLang] = botEditorPrompt.value;
      botEditorStatus.style.display = "inline";
      await loadBotsList();
    } catch (err) {
      botEditorError.textContent = err.message;
      botEditorError.style.display = "block";
    } finally {
      botEditorSavePromptBtn.disabled = false;
    }
  });

  addBotBtn.addEventListener("click", () => {
    addBotError.style.display = "none";
    addBotForm.reset();
    addBotOverlay.classList.remove("hidden");
  });
  addBotClose.addEventListener("click", () => addBotOverlay.classList.add("hidden"));
  addBotOverlay.addEventListener("click", (e) => { if (e.target === addBotOverlay) addBotOverlay.classList.add("hidden"); });

  addBotForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    addBotError.style.display = "none";
    try {
      const created = await fetchJSON("/api/bots", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: document.getElementById("admin-new-bot-name").value,
        }),
      });
      addBotOverlay.classList.add("hidden");
      await loadBotsList();
      await openBotEditor(created.id);
    } catch (err) {
      addBotError.textContent = err.message;
      addBotError.style.display = "block";
    }
  });

  // ---- call logs tab -------------------------------------------------------------

  const callLogsBody = document.getElementById("admin-calllogs-body");
  const callLogsEmpty = document.getElementById("calllogs-empty");
  const callLogDetailTitle = document.getElementById("calllog-detail-title");
  const callLogDetailClose = document.getElementById("calllog-detail-close");
  const callLogDetailMeta = document.getElementById("calllog-detail-meta");
  const callLogDetailTranscript = document.getElementById("calllog-detail-transcript");

  function formatDuration(seconds) {
    if (seconds == null) return "—";
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  function formatCost(inr) {
    return inr == null ? "—" : `₹${inr.toFixed(2)}`;
  }

  function recordingUrl(c) {
    return `/api/call-logs/${encodeURIComponent(c.dir)}/recording`;
  }

  // A download link rather than a button: the browser handles the save itself
  // (and honours the Content-Disposition filename the API sends), and the
  // admin can middle-click to open it in a tab and listen without saving.
  function recordingCell(c) {
    if (!c.has_recording) {
      const span = document.createElement("span");
      span.className = "muted";
      span.textContent = "—";
      span.title = c.complete ? "no audio captured for this call" : "call still in progress";
      return span;
    }
    const link = document.createElement("a");
    link.className = "btn";
    link.href = recordingUrl(c);
    link.setAttribute("download", "");
    link.textContent = "Download";
    link.title = c.recording_s3_key
      ? `Archived to S3: ${c.recording_s3_key}`
      : "Saved on the server (not archived to S3)";
    return link;
  }

  function renderCallLogRow(c) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${adminEscapeHtml(c.username || "unknown")}</td>
      <td>${formatUserDate(c.started_at)}</td>
      <td>${adminEscapeHtml(c.language ? c.language.toUpperCase() : "—")}</td>
      <td>${adminEscapeHtml(c.tts_tier || "—")}</td>
      <td>${c.complete ? formatDuration(c.duration_s) : "in progress"}</td>
      <td>${c.complete ? formatCost(c.cost_inr) : "—"}</td>
      <td class="calllog-recording"></td>
      <td><div class="row-actions"></div></td>
    `;
    tr.querySelector(".calllog-recording").appendChild(recordingCell(c));
    const actions = tr.querySelector(".row-actions");
    const viewBtn = document.createElement("button");
    viewBtn.className = "btn";
    viewBtn.textContent = "View";
    viewBtn.addEventListener("click", () => openCallLogDetail(c));
    actions.appendChild(viewBtn);
    return tr;
  }

  async function loadCallLogs() {
    callLogsBody.innerHTML = "";
    try {
      const logs = await fetchJSON("/api/call-logs");
      callLogsEmpty.style.display = logs.length ? "none" : "block";
      for (const c of logs) callLogsBody.appendChild(renderCallLogRow(c));
    } catch (err) {
      callLogsBody.innerHTML = `<tr><td colspan="8" class="form-error">failed to load call logs: ${err.message}</td></tr>`;
    }
  }

  async function openCallLogDetail(c) {
    callLogDetailTitle.textContent = `Call with ${c.username || "unknown"}`;
    callLogDetailMeta.textContent = "Loading...";
    callLogDetailTranscript.innerHTML = "";
    callLogDetailOverlay.classList.remove("hidden");
    try {
      const detail = await fetchJSON(`/api/call-logs/${encodeURIComponent(c.dir)}`);
      const parts = [
        formatUserDate(detail.started_at),
        detail.language ? detail.language.toUpperCase() : "—",
        detail.tts_tier || "—",
        detail.complete ? `${formatDuration(detail.duration_s)} · ${formatCost(detail.cost_inr)}` : "call still in progress",
      ];
      callLogDetailMeta.textContent = parts.join(" · ");
      if (detail.has_recording) {
        const player = document.createElement("audio");
        player.controls = true;
        player.preload = "none"; // don't pull the whole WAV just for opening the dialog
        player.src = recordingUrl(detail);
        player.style.width = "100%";
        player.style.margin = "0 0 12px";
        callLogDetailTranscript.appendChild(player);
      }
      if (!detail.transcript.length) {
        callLogDetailTranscript.insertAdjacentHTML(
          "beforeend",
          '<p class="empty-state">No conversation recorded (call ended before anyone spoke).</p>'
        );
        return;
      }
      for (const msg of detail.transcript) {
        const div = document.createElement("div");
        div.className = `msg ${msg.role}` + (msg.interrupted ? " interrupted" : "");
        const who = document.createElement("span");
        who.className = "who";
        who.textContent = msg.role === "user" ? "Caller" : "Agent";
        div.appendChild(who);
        div.appendChild(document.createTextNode(msg.text));
        if (msg.interrupted) {
          const tag = document.createElement("span");
          tag.className = "badge interrupted-tag";
          tag.textContent = "Interrupted";
          div.appendChild(tag);
        }
        callLogDetailTranscript.appendChild(div);
      }
    } catch (err) {
      callLogDetailMeta.textContent = "";
      callLogDetailTranscript.innerHTML = `<p class="form-error">failed to load transcript: ${err.message}</p>`;
    }
  }

  callLogDetailClose.addEventListener("click", () => callLogDetailOverlay.classList.add("hidden"));
  callLogDetailOverlay.addEventListener("click", (e) => {
    if (e.target === callLogDetailOverlay) callLogDetailOverlay.classList.add("hidden");
  });

  // ---- initial load ------------------------------------------------------------
  // this page IS the admin panel (no open/close) - load all three tabs' data as
  // soon as it renders. bots must load first - loadUsers()'s per-row bot dropdown
  // reads the allBots cache.
  await loadBotsList();
  await loadUsers();
  await loadCallLogs();
})();
