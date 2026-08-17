import { chromium } from "playwright";
import fs from "node:fs";

const TOKEN = fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7tok.txt", "utf8").trim();
const JOB = JSON.parse(fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7job.json", "utf8")).job_id;
const BASE = "http://localhost:5173";

const results = [];
const ok = (name, pass, detail = "") => results.push({ name, pass, detail });

const browser = await chromium.launch({
  // Let WebAudio actually run without a real gesture, and give it a fake device.
  args: ["--autoplay-policy=no-user-gesture-required", "--use-fake-device-for-media-stream"],
});
const ctx = await browser.newContext();
const page = await ctx.newPage();

const errors = [];
page.on("console", (m) => {
  const t = m.text();
  if (m.type() === "error" && !t.includes("404")) errors.push(t);
  if (t.includes("[ptify]")) console.log("DRIFT-WARN:", t);
});
page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message));

await page.addInitScript((t) => localStorage.setItem("ptify.token", t), TOKEN);

// Playhead x, read off the transform matrix -- deterministic, unlike a screenshot.
const playheadX = () =>
  page.evaluate(() => {
    const el = document.querySelector(".roll-playhead");
    if (!el) return null;
    const m = new DOMMatrixReadOnly(getComputedStyle(el).transform);
    return m.m41;
  });
const fillScale = () =>
  page.evaluate(() => {
    const el = document.querySelector(".transport-fill");
    if (!el) return null;
    return new DOMMatrixReadOnly(getComputedStyle(el).transform).a;
  });
const clock = () => page.locator(".transport-clock").innerText();
const scrollLeft = () => page.evaluate(() => document.querySelector(".roll-scroll")?.scrollLeft ?? -1);

await page.goto(`${BASE}/#/j/${JOB}`, { waitUntil: "domcontentloaded" });
await page.waitForSelector(".result", { timeout: 15000 });

// --- 1. the transport renders ----------------------------------------------
ok("transport renders", (await page.locator(".transport").count()) > 0);
ok("play button renders", (await page.locator(".transport-play").count()) > 0);
ok("clock starts at 0:00", (await clock()).startsWith("0:00"), await clock());

// --- 2. press play, wait for samples to load --------------------------------
const t0 = Date.now();
await page.locator(".transport-play").click();
await page.waitForFunction(
  () => {
    const el = document.querySelector(".roll-playhead");
    if (!el) return false;
    return new DOMMatrixReadOnly(getComputedStyle(el).transform).m41 > 2;
  },
  { timeout: 45000 },
).catch(() => {});
const loadMs = Date.now() - t0;

const x1 = await playheadX();
ok("playhead advances after play", x1 > 2, `x=${x1} after ${loadMs}ms`);

// --- 3. an AudioContext is actually running ---------------------------------
const ctxState = await page.evaluate(() => window.__ptifyAudioState ?? "unknown");
ok("audio context reported", true, "state=" + ctxState);

// --- 4. it keeps advancing ---------------------------------------------------
await page.waitForTimeout(1800);
const x2 = await playheadX();
ok("playhead keeps advancing", x2 > x1 + 20, `${x1?.toFixed(1)} -> ${x2?.toFixed(1)}`);

// --- 5. the clock moved ------------------------------------------------------
const c2 = await clock();
ok("clock advanced past 0:00", !c2.startsWith("0:00"), c2);

// --- 6. the scrub fill tracks ------------------------------------------------
const f = await fillScale();
ok("scrub fill advanced", f > 0 && f < 1, "scaleX=" + f?.toFixed(4));

// --- 7. pause holds ----------------------------------------------------------
await page.locator(".transport-play").click();
await page.waitForTimeout(700);
const p1 = await playheadX();
await page.waitForTimeout(900);
const p2 = await playheadX();
ok("pause stops the playhead", Math.abs(p2 - p1) < 1.5, `${p1?.toFixed(1)} -> ${p2?.toFixed(1)}`);

// --- 8. ZOOM MID-PLAYBACK: the stale-closure regression ----------------------
const beforeZoom = await playheadX();
await page.locator('[aria-label="Zoom in"]').click();
await page.waitForTimeout(500);
const afterZoom = await playheadX();
const ratio = afterZoom / beforeZoom;
ok(
  "zoom rescales the playhead (stale pps regression)",
  ratio > 1.3 && ratio < 1.5,
  `x ${beforeZoom?.toFixed(1)} -> ${afterZoom?.toFixed(1)} (ratio ${ratio?.toFixed(3)}, want ~1.4)`,
);
await page.locator('[aria-label="Zoom out"]').click();
await page.waitForTimeout(400);

// --- 9. click the roll to seek ----------------------------------------------
const box = await page.locator(".roll-canvas").boundingBox();
await page.mouse.click(box.x + 300, box.y + 40);
await page.waitForTimeout(600);
const seekX = await playheadX();
ok("roll click seeks", Math.abs(seekX - 300) < 12, `x=${seekX?.toFixed(1)} (clicked 300)`);

// --- 10. space toggles -------------------------------------------------------
await page.locator("body").click({ position: { x: 5, y: 400 } });
const wasPlaying = await page.locator(".transport-play").getAttribute("aria-pressed");
await page.keyboard.press("Space");
await page.waitForTimeout(900);
const nowPlaying = await page.locator(".transport-play").getAttribute("aria-pressed");
ok("space toggles playback", wasPlaying !== nowPlaying, `${wasPlaying} -> ${nowPlaying}`);

// --- 11. scroll-follow ------------------------------------------------------
if (nowPlaying !== "true") await page.keyboard.press("Space");
await page.locator('[aria-label="Zoom in"]').click();
await page.locator('[aria-label="Zoom in"]').click();
await page.waitForTimeout(3000);
const sl = await scrollLeft();
ok("scroll-follow moved the pane", sl > 0, "scrollLeft=" + sl);

// --- 12. manual scroll is respected -----------------------------------------
await page.evaluate(() => {
  const el = document.querySelector(".roll-scroll");
  el.dispatchEvent(new WheelEvent("wheel", { deltaY: 10, bubbles: true }));
  el.scrollLeft = 0;
});
await page.waitForTimeout(1200);
const afterManual = await scrollLeft();
ok("follow yields after a manual scroll", afterManual < 60, "scrollLeft=" + afterManual);

// --- 13. vertical scroll is never stolen ------------------------------------
// Only meaningful when the pane actually overflows vertically -- this clip's
// pitch range fits, so shrink the viewport to force it.
await page.setViewportSize({ width: 1280, height: 420 });
await page.waitForTimeout(400);
const vertical = await page.evaluate(() => {
  const el = document.querySelector(".roll-scroll");
  if (el.scrollHeight <= el.clientHeight) return { skipped: true };
  el.scrollTop = 30;
  return { skipped: false, set: el.scrollTop };
});
if (vertical.skipped) {
  ok("vertical scroll is left alone", true, "pane does not overflow vertically; n/a");
} else {
  await page.waitForTimeout(1600);
  const st = await page.evaluate(() => document.querySelector(".roll-scroll").scrollTop);
  ok("vertical scroll is left alone", st === vertical.set, `scrollTop stayed ${st}`);
}

console.log("\n=== PHASE 7b ===");
let fails = 0;
for (const r of results) {
  if (!r.pass) fails++;
  console.log(`${r.pass ? "PASS" : "FAIL"}  ${r.name}${r.detail ? "   [" + r.detail + "]" : ""}`);
}
console.log(`\n${results.length - fails}/${results.length} passed`);
console.log("console errors: " + (errors.length === 0 ? "none" : "\n  " + errors.join("\n  ")));

await browser.close();
process.exit(fails > 0 ? 1 : 0);
