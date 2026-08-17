import { chromium } from "playwright";
import fs from "node:fs";

const TOKEN = fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7tok.txt", "utf8").trim();
const JOB = JSON.parse(fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7job.json", "utf8")).job_id;
const BASE = "http://localhost:5173";
const OUT = "C:/Users/SeanN/LivePianoSynthesizer/var/shots";
fs.mkdirSync(OUT, { recursive: true });

const results = [];
const ok = (n, p, d = "") => results.push({ n, p, d });

const browser = await chromium.launch();

// ============ default motion =============================================
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();
const errors = [];
page.on("console", (m) => { if (m.type() === "error" && !m.text().includes("404")) errors.push(m.text()); });
page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message));
await page.addInitScript((t) => localStorage.setItem("ptify.token", t), TOKEN);

// --- M1 curtain -------------------------------------------------------------
await page.goto(`${BASE}/#/`, { waitUntil: "commit" });
const sawCurtain = await page.waitForSelector(".boot", { timeout: 4000 }).then(() => true).catch(() => false);
ok("M1 curtain appears on boot", sawCurtain);
if (sawCurtain) {
  const mark = await page.locator(".boot-mark").count();
  ok("M1 curtain shows the brand mark (not a blank panel)", mark > 0);
  await page.screenshot({ path: `${OUT}/10-curtain.png` });

  // The panel must cover the viewport for the WHOLE lift. A curtain sized to
  // exactly 100vh uncovers the app the moment it starts moving, so the page
  // shows through beneath it -- which reads as a flash, not a transition.
  const covers = await page.evaluate(() => {
    const el = document.querySelector(".boot");
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { h: r.height, vh: window.innerHeight, top: r.top };
  });
  ok(
    "M1 curtain is taller than the viewport (nothing shows through mid-lift)",
    covers && covers.h > covers.vh * 1.5,
    covers ? `panel ${Math.round(covers.h)}px vs viewport ${covers.vh}px` : "gone",
  );
  // It must lift, then unmount.
  const gone = await page.waitForFunction(() => !document.querySelector(".boot"), { timeout: 6000 })
    .then(() => true).catch(() => false);
  ok("M1 curtain lifts away and unmounts", gone);
}

// --- M2 word stagger ---------------------------------------------------------
// The curtain covers the viewport for its lift, so every later navigation has
// to wait it out before anything counts as visible.
const settle = async (p, sel) => {
  await p.waitForFunction(() => !document.querySelector(".boot"), { timeout: 8000 }).catch(() => {});
  await p.waitForSelector(sel, { timeout: 12000 });
  await p.waitForTimeout(1400);
};

await page.goto(`${BASE}/#/`, { waitUntil: "domcontentloaded" });
await settle(page, ".upload");
const words = await page.locator(".upload .reveal .rv-word").count();
ok("M2 heading is split per word", words === 3, `"Choose a recording." -> ${words} words`);

const delays = await page.evaluate(() => {
  const els = [...document.querySelectorAll(".upload .reveal .rv-inner")];
  return els.map((e) => getComputedStyle(e).transitionDelay);
});
ok("M2 delays step by the stagger token", delays.join(",") === "0s,0.042s,0.084s", delays.join(" "));

const settled = await page.evaluate(() => {
  const els = [...document.querySelectorAll(".upload .reveal .rv-inner")];
  return els.every((e) => +getComputedStyle(e).opacity > 0.95);
});
ok("M2 words end up visible", settled);

// The heading must still read as one string to a screen reader.
const label = await page.locator(".upload .reveal").getAttribute("aria-label");
ok("M2 heading keeps an accessible label", label === "Choose a recording.", String(label));

// --- the &nbsp;/<em> heading survives ----------------------------------------
// A SEPARATE context: this page's addInitScript re-seeds the token on every
// navigation, so clearing localStorage here would just sign straight back in.
const actx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const anon = await actx.newPage();
anon.on("pageerror", (e) => errors.push("ANON PAGEERROR: " + e.message));
await anon.goto(`${BASE}/#/sign-in`, { waitUntil: "domcontentloaded" });
await settle(anon, ".auth");
const emCount = await anon.locator(".auth .reveal em").count();
const authLabel = await anon.locator(".auth .reveal").getAttribute("aria-label");
// innerText reports no whitespace between inline-block spans even when the
// visual gap is real, so assert the RENDERED gap instead of a string.
const gap = await anon.evaluate(() => {
  const w = [...document.querySelectorAll(".auth .reveal .rv-word")];
  // Two words known to sit on the same line ("Notes" / "you" may wrap), so
  // measure every adjacent pair that shares a baseline and take the smallest.
  let min = Infinity;
  for (let i = 1; i < w.length; i++) {
    const a = w[i - 1].getBoundingClientRect();
    const b = w[i].getBoundingClientRect();
    if (Math.abs(a.top - b.top) < 4) min = Math.min(min, b.left - a.right);
  }
  return min === Infinity ? null : min;
});
ok("M2 markup survives the split (<em> intact)", emCount === 1, `${emCount} <em>`);
ok("M2 words are visually separated", gap != null && gap > 4, `min word gap = ${gap?.toFixed(1)}px`);

// A word gap made of margin survives a line break and indents every wrapped
// line -- a ragged left edge that looks like a layout bug rather than a choice.
const lefts = await anon.evaluate(() => {
  const rows = new Map();
  for (const w of document.querySelectorAll(".auth .reveal .rv-word")) {
    const r = w.getBoundingClientRect();
    const key = Math.round(r.top);
    if (!rows.has(key) || r.left < rows.get(key)) rows.set(key, r.left);
  }
  return [...rows.values()].map((v) => Math.round(v));
});
ok(
  "M2 wrapped lines share a left edge",
  lefts.length > 0 && Math.max(...lefts) - Math.min(...lefts) < 2,
  `line starts: ${lefts.join(", ")}`,
);
ok("M2 nbsp heading has the right label", authLabel === "Notes you can read, edit, and trust.", String(authLabel));
await anon.screenshot({ path: `${OUT}/11-auth-stagger.png` });
await actx.close();

// --- M3 enter-stagger --------------------------------------------------------
await page.addInitScript((t) => localStorage.setItem("ptify.token", t), TOKEN);
await page.evaluate((t) => localStorage.setItem("ptify.token", t), TOKEN);
await page.goto(`${BASE}/#/history`, { waitUntil: "domcontentloaded" });
await settle(page, ".history");
const staggered = await page.evaluate(() => {
  const kids = [...document.querySelectorAll(".history.enter-stagger > *")];
  return kids.map((k) => getComputedStyle(k).animationDelay);
});
ok("M3 children stagger", staggered.length >= 2 && staggered[0] === "0s" && staggered[1] === "0.042s",
   staggered.join(" "));

// --- M4 arrow slide ----------------------------------------------------------
const rows = await page.locator(".job-row:not(.is-inert)").count();
if (rows > 0) {
  const before = await page.evaluate(() => {
    const a = document.querySelector(".job-row:not(.is-inert) .job-arrow");
    return { op: getComputedStyle(a).opacity, dur: getComputedStyle(a).transitionDuration };
  });
  await page.locator(".job-row:not(.is-inert)").first().hover();
  await page.waitForTimeout(700);
  const after = await page.evaluate(() => {
    const r = document.querySelector(".job-row:not(.is-inert)");
    const a = r.querySelector(".job-arrow");
    const w = r.querySelector(".job-when");
    return {
      op: getComputedStyle(a).opacity,
      arrowDur: getComputedStyle(a).transitionDuration,
      textDur: getComputedStyle(w).transitionDuration,
      textX: new DOMMatrixReadOnly(getComputedStyle(w).transform).m41,
    };
  });
  ok("M4 arrow hidden until hover", +before.op < 0.05, "opacity " + before.op);
  ok("M4 arrow fades in on hover", +after.op > 0.9, "opacity " + after.op);
  ok("M4 text slides on hover", after.textX > 4, `translateX ${after.textX?.toFixed(1)}px`);
  ok("M4 the two durations differ (arrow 300ms, slide 400ms)",
     after.arrowDur.startsWith("0.3") && after.textDur.startsWith("0.4"),
     `arrow ${after.arrowDur} / text ${after.textDur}`);
  await page.screenshot({ path: `${OUT}/12-history-hover.png` });
} else {
  ok("M4 arrow slide", false, "no openable job rows to hover");
}

// --- M6 page turn ------------------------------------------------------------
await page.goto(`${BASE}/#/j/${JOB}/sheet`, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => !document.querySelector(".boot"), { timeout: 8000 }).catch(() => {});
await page.waitForSelector(".sheet-svg", { timeout: 20000 });
const anim = await page.evaluate(() => {
  const el = document.querySelector(".sheet-svg");
  const s = getComputedStyle(el);
  return { name: s.animationName, dur: s.animationDuration };
});
ok("M6 sheet page animates in", anim.name === "pageTurn", `${anim.name} ${anim.dur}`);
await page.screenshot({ path: `${OUT}/13-sheet.png` });

// --- M5 press feedback --------------------------------------------------------
// Format chips live on step 2 of the upload flow, which needs a file first.
await page.goto(`${BASE}/#/`, { waitUntil: "domcontentloaded" });
await settle(page, ".drop");
await page.locator('input[type="file"]').setInputFiles(
  "C:/Users/SeanN/LivePianoSynthesizer/var/clip25.wav",
);
await settle(page, ".chip");
const tapDur = await page.evaluate(() => {
  const c = document.querySelector(".chip");
  const s = getComputedStyle(c);
  return s.transitionDuration;
});
ok("M5 chips have tap feedback wired", tapDur.includes("0.09"), tapDur);

await ctx.close();

// ============ reduced motion ==============================================
const rctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
const rp = await rctx.newPage();
rp.on("pageerror", (e) => errors.push("RM PAGEERROR: " + e.message));
await rp.addInitScript((t) => localStorage.setItem("ptify.token", t), TOKEN);
await rp.goto(`${BASE}/#/`, { waitUntil: "domcontentloaded" });
await settle(rp, ".upload");

const rm = await rp.evaluate(() => {
  const els = [...document.querySelectorAll(".reveal .rv-inner")];
  return {
    delays: els.map((e) => getComputedStyle(e).transitionDelay),
    opacities: els.map((e) => +getComputedStyle(e).opacity),
  };
});
ok("RM: stagger delays are zeroed", rm.delays.every((d) => d === "0s"), rm.delays.join(" "));
ok("RM: words are visible, not stuck at opacity 0", rm.opacities.every((o) => o > 0.95),
   rm.opacities.join(" "));
await rp.screenshot({ path: `${OUT}/14-reduced-motion.png` });
await rctx.close();

console.log("\n=== PHASE 7c MOTION ===");
let fails = 0;
for (const r of results) { if (!r.p) fails++; console.log(`${r.p ? "PASS" : "FAIL"}  ${r.n}${r.d ? "   [" + r.d + "]" : ""}`); }
console.log(`\n${results.length - fails}/${results.length} passed`);
console.log("console errors: " + (errors.length ? "\n  " + errors.join("\n  ") : "none"));
await browser.close();
process.exit(fails ? 1 : 0);
