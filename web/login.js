const form = document.getElementById("login-form");
const errorEl = document.getElementById("login-error");
const submitBtn = document.getElementById("login-submit");
const passwordInput = document.getElementById("password");
const passwordToggle = document.getElementById("password-toggle");

passwordToggle.addEventListener("click", () => {
  const showing = passwordInput.type === "text";
  passwordInput.type = showing ? "password" : "text";
  passwordToggle.setAttribute("aria-label", showing ? "Show password" : "Hide password");
});

const params = new URLSearchParams(location.search);
const redirectTo = params.get("next") || "/";

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.style.display = "none";
  submitBtn.disabled = true;
  submitBtn.textContent = "Signing in...";
  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: document.getElementById("username").value,
        password: document.getElementById("password").value,
        remember: document.getElementById("remember").checked,
      }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || "login failed");
    }
    location.href = redirectTo;
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.style.display = "block";
    submitBtn.disabled = false;
    submitBtn.textContent = "Sign in";
  }
});
