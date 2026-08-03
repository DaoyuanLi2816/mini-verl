(() => {
  const banner = document.querySelector(".docs-channel");
  const selector = document.getElementById("docs-version-selector");
  const label = document.getElementById("docs-channel-label");
  if (!banner || !selector || !label) return;

  const pathname = window.location.pathname;
  const devMarker = "/dev/";
  const isDev = pathname.includes(devMarker);
  const scriptPath = document.currentScript
    ? new URL(document.currentScript.src).pathname
    : pathname;
  const assetsMarker = "/assets/javascripts/";
  let base = scriptPath.includes(assetsMarker)
    ? `${scriptPath.slice(0, scriptPath.indexOf(assetsMarker))}/`
    : pathname.replace(/[^/]*$/, "");
  if (base.endsWith(devMarker)) base = base.slice(0, -devMarker.length + 1);
  const stableVersion = banner.dataset.stableVersion;
  const devVersion = banner.dataset.devVersion;

  selector.value = isDev ? "dev" : "stable";
  label.textContent = isDev
    ? `Development documentation · ${devVersion}`
    : `Stable documentation · ${stableVersion}`;

  selector.addEventListener("change", () => {
    window.location.assign(selector.value === "dev" ? `${base}dev/` : base);
  });
})();
