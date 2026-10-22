// Shared shell behavior: active nav highlighting + a small fetch helper.

(function highlightActiveNav() {
  const here = location.pathname;
  document.querySelectorAll(".nav-item").forEach((el) => {
    if (el.getAttribute("href") === here) el.classList.add("active");
  });
})();

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

// Resolves to the logged-in user, populates the topbar chip + admin-only nav
// items, and wires Logout. Redirects to /login if the session isn't valid -
// a client-side backstop; the real gate is the server-side page routes.
window.currentUserPromise = (async () => {
  let user;
  try {
    user = await fetchJSON("/api/me");
  } catch (err) {
    location.href = "/login?next=" + encodeURIComponent(location.pathname);
    return null;
  }

  if (user.role === "admin") {
    document.querySelectorAll("[data-admin-only]").forEach((el) => { el.style.display = ""; });
  }
  const nameEl = document.getElementById("topbar-username");
  if (nameEl) {
    // omit the "(role)" suffix when it's redundant with the username itself
    // (e.g. the seed account is literally named "admin" with role "admin")
    nameEl.textContent = user.username.toLowerCase() === user.role.toLowerCase()
      ? user.username
      : `${user.username} (${user.role})`;
  }
  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", async () => {
      await fetch("/api/logout", { method: "POST" });
      location.href = "/login";
    });
  }
  return user;
})();
