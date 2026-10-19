"""Browser-based visual + contrast audit for FinSIGHT (optional developer tool).

Starts the app, drives it with headless Chromium through every page in BOTH
themes, takes screenshots and measures text contrast (WCAG ratio) for every
visible piece of text, including chart labels. It reports anything below 4.5:1
(3:1 for large text).

Usage (from the project root):
    pip install playwright && playwright install chromium
    python tools/visual_check.py            # writes docs/screenshots/*.png and prints a report

Exit code is 1 when a contrast failure or a runtime error is found.
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
PORT = 8765
URL = f"http://localhost:{PORT}/"

PAGES = [("Overview Dashboard", "dashboard"), ("Test Dataset Explorer", "explorer"), ("New Customer Prediction", "predict"),
         ("Model Comparison", "compare"), ("Performance & Insights", "insights"), ("Retention Insights", "retention"),
         ("Prediction History", "history")]

AUDIT_JS = r"""
() => {
  const parse = (c) => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null;
    const p = m[1].split(',').map(x => parseFloat(x)); return {r:p[0], g:p[1], b:p[2], a: p.length > 3 ? p[3] : 1}; };
  const lum = (c) => { const f = v => { v/=255; return v<=0.03928 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4); };
    return 0.2126*f(c.r)+0.7152*f(c.g)+0.0722*f(c.b); };
  const over = (top, bottom) => ({ r: top.r*top.a + bottom.r*(1-top.a), g: top.g*top.a + bottom.g*(1-top.a), b: top.b*top.a + bottom.b*(1-top.a), a: 1 });
  const bgOf = (el) => { const layers = []; let n = el;
    while (n && n.nodeType === 1) { if (!(n instanceof HTMLElement)) { n = n.parentElement; continue; } const c = parse(getComputedStyle(n).backgroundColor); if (c && c.a > 0) { layers.push(c); if (c.a >= 0.99) break; } n = n.parentElement; }
    let base = {r:255,g:255,b:255,a:1}; if (!layers.length || layers[layers.length-1].a < 0.99) { const b = parse(getComputedStyle(document.querySelector('.stApp')).backgroundColor); if (b && b.a > 0.5) base = b; }
    for (let i = layers.length-1; i >= 0; i--) base = over(layers[i], base); return base; };
  const results = []; const seen = new Set();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const text = node.textContent.trim(); if (!text) continue;
    const el = node.parentElement; if (!el) continue;
    if (el.closest('script,style,noscript,[aria-hidden="true"] svg path')) continue;
    const cs = getComputedStyle(el); const r = el.getBoundingClientRect();
    if (cs.visibility === 'hidden' || cs.display === 'none' || r.width < 2 || r.height < 2) continue;
    if (el.closest('[data-testid="stToolbar"],[data-testid="stDecoration"]')) continue;
    // data labels drawn ON coloured chart marks are verified numerically in tests/test_theme.py
    if (el instanceof SVGElement && el.closest('.slicetext,.textpoint,.bartext,.heatmaplayer,.annotation')) continue;
    let fgc = parse(cs.color); if (el instanceof SVGElement) fgc = parse(cs.fill) || fgc;
    if (!fgc) continue;
    let eff = 1; let a = el; while (a && a.nodeType === 1) { eff *= parseFloat(getComputedStyle(a).opacity); a = a.parentElement; }
    const bg = bgOf(el); const fg = over({...fgc, a: fgc.a * Math.min(1, eff)}, bg);
    const ratio = (Math.max(lum(fg), lum(bg)) + 0.05) / (Math.min(lum(fg), lum(bg)) + 0.05);
    const px = parseFloat(cs.fontSize); const bold = parseInt(cs.fontWeight) >= 700;
    const large = px >= 24 || (px >= 18.66 && bold);
    const need = large ? 3 : 4.5;
    const key = text.slice(0,40) + '|' + ratio.toFixed(2);
    if (ratio < need && !seen.has(key)) { seen.add(key); results.push({text: text.slice(0,60), ratio: +ratio.toFixed(2), need, color: cs.color, tag: el.tagName.toLowerCase()}); }
  }
  return results;
}
"""


def wait_for_server(timeout: float = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(URL, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Streamlit did not start")


def settle(page, ms: int = 1800) -> None:
    page.wait_for_timeout(ms)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60000)


def set_theme(page, mode: str) -> None:
    want_dark = mode == "dark"
    toggle = page.get_by_role("switch", name="Dark mode")
    if toggle.is_checked() != want_dark:
        page.locator("label", has=page.locator('input[role="switch"]')).first.click()
        settle(page, 2200)
    assert toggle.is_checked() == want_dark, "theme switch did not take effect"


def go(page, label: str) -> None:
    page.get_by_test_id("stPageLink-NavLink").filter(has_text=label).first.click()
    settle(page)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                               "--server.port", str(PORT)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failures: list[str] = []
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 2600})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(URL)
            page.wait_for_selector("text=Customer Churn Intelligence", timeout=90000)
            settle(page, 2500)

            for mode in ("light", "dark"):
                set_theme(page, mode)
                for label, slug in PAGES:
                    go(page, label)
                    # drive the interactive pages so result panels are audited too
                    if slug == "explorer":
                        page.get_by_role("button", name="Generate Random Test Record").click(); settle(page, 3000)
                    if slug == "predict":
                        page.get_by_role("button", name="Predict Customer Churn").click(); settle(page, 3500)
                        page.locator('[data-testid="stSelectbox"]').first.click(); page.wait_for_timeout(700)   # open a dropdown
                        for item in page.evaluate(AUDIT_JS):
                            failures.append(f"[{mode}] {label} (dropdown open): {item}")
                        page.screenshot(path=str(OUT / f"{mode}_predict_dropdown-open.png"))
                        page.keyboard.press("Escape"); page.wait_for_timeout(400)
                    if slug == "compare":
                        page.get_by_role("tab", name="Overall performance").click(); settle(page, 1500)
                        page.get_by_role("tab", name="Individual customer").click(); settle(page, 1200)
                    if slug == "insights":
                        for tab in ("Churn distribution", "Confusion matrices", "ROC curves", "Precision-recall", "Feature importance"):
                            page.get_by_role("tab", name=tab).click(); settle(page, 1800)
                            bad = page.evaluate(AUDIT_JS)
                            for item in bad:
                                failures.append(f"[{mode}] {label}/{tab}: {item}")
                            page.screenshot(path=str(OUT / f"{mode}_{slug}_{tab.lower().replace(' ', '-')}.png"), full_page=True)
                        page.get_by_role("tab", name="Performance").first.click(); settle(page, 1200)
                    page.screenshot(path=str(OUT / f"{mode}_{slug}.png"), full_page=True)
                    for item in page.evaluate(AUDIT_JS):
                        failures.append(f"[{mode}] {label}: {item}")
                    alert = page.locator('[data-testid="stException"]').count()
                    if alert:
                        failures.append(f"[{mode}] {label}: Streamlit exception displayed")
            # the theme must survive reruns: it was dark at the end of the loop; confirm after navigating once more
            go(page, "Overview Dashboard")
            if not page.get_by_role("switch", name="Dark mode").is_checked():
                failures.append("theme did not persist across page navigation")
            failures += [f"browser error: {e}" for e in errors]
            browser.close()
    finally:
        server.terminate()
    print(f"Screenshots written to {OUT}")
    if failures:
        print(f"\n{len(failures)} problem(s):")
        for f in failures:
            print(" -", f)
        return 1
    print("No contrast failures, runtime errors or theme persistence problems found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
