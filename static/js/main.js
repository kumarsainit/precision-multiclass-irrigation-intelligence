function showToast(message, isError = false) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = message;
  toast.className = isError ? "show error" : "show";
  setTimeout(() => { toast.className = ""; }, 3500);
}

function badgeClass(level) {
  const value = (level || "").toLowerCase();
  if (value === "none") return "badge badge-none";
  if (value === "low") return "badge badge-low";
  if (value === "medium") return "badge badge-medium";
  if (value === "high") return "badge badge-high";
  return "badge";
}

function fmtDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString("en-US", {
    month: "short", day: "numeric", year: "numeric",
    hour: "2-digit", minute: "2-digit"
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    const output = document.getElementById(`${slider.id}_val`);
    if (!output) return;
    output.textContent = slider.value;
    slider.addEventListener("input", () => { output.textContent = slider.value; });
  });
});
