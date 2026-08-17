import { chromium } from "playwright";
import fs from "node:fs";

const TOKEN = fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7tok.txt", "utf8").trim();
const CLIP = "C:/Users/SeanN/LivePianoSynthesizer/var/clip25.wav";
const BASE = "http://localhost:5173";
const OUT = "C:/Users/SeanN/LivePianoSynthesizer/var/shots";
fs.mkdirSync(OUT, { recursive: true });

const results = [];
const ok = (n, p, d = "") => results.push({ n, p, d });

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();
const errors = [];
page.on("console", (m) => { if (m.type() === "error" && !m.text().includes("404")) errors.push(m.text()); });
page.on("pageerror", (e) => errors.push("PAGEERROR: " + e.message));
await page.addInitScript((t) => localStorage.setItem("ptify.token", t), TOKEN);

const settle = async (sel) => {
  await page.waitForFunction(() => !document.querySelector(".boot"), { timeout: 9000 }).catch(() => {});
  await page.waitForSelector(sel, { timeout: 12000 });
  await page.waitForTimeout(900);
};

// --- step 1: only the recording ---------------------------------------------
await page.goto(`${BASE}/#/`, { waitUntil: "domcontentloaded" });
await settle(".step-file");
ok("step 1 shows the dropzone", (await page.locator(".drop").count()) > 0);
ok("step 1 hides engine choices", (await page.locator(".engine").count()) === 0);
ok("step 1 hides format chips", (await page.locator(".chip").count()) === 0);
ok("step 1 hides the metadata fields", (await page.locator(".details-card").count()) === 0);
ok("progress rail has 3 steps", (await page.locator(".step-item").count()) === 3);
ok("step 1 is current", (await page.locator(".step-item.is-current .step-label").innerText()) === "Recording");
await page.screenshot({ path: `${OUT}/s1-file.png` });

// --- choosing a file advances by itself --------------------------------------
await page.locator('input[type="file"]').setInputFiles(CLIP);
const advanced = await page.waitForFunction(
  () => location.hash.includes("/new/output"),
  { timeout: 6000 },
).then(() => true).catch(() => false);
ok("choosing a file advances on its own", advanced, page.url().split("#")[1] ?? "");
await settle(".step-output");

// --- step 2: output ----------------------------------------------------------
ok("step 2 shows the engines", (await page.locator(".engine").count()) >= 1);
ok("step 2 shows the format chips", (await page.locator(".chip").count()) === 5);
ok("step 2 hides the dropzone", (await page.locator(".drop").count()) === 0);
const staggered = await page.evaluate(() =>
  [...document.querySelectorAll(".engine")].map((e) => getComputedStyle(e).animationDelay));
ok("step 2 children stagger in", staggered[0] === "0.12s" && staggered[1] === "0.162s", staggered.join(" "));
await page.screenshot({ path: `${OUT}/s2-output.png` });

// --- step 3: details ---------------------------------------------------------
await page.locator(".step-nav .btn:not(.btn-ghost)").click();
await settle(".step-details");
ok("step 3 shows the metadata fields", (await page.locator(".details-card .field-input").count()) === 4);
ok("step 3 shows a summary", (await page.locator(".summary-card").count()) > 0);
const sumFile = await page.locator(".summary-file").innerText();
ok("summary names the chosen recording", sumFile.includes("clip25"), sumFile);
const sumChips = await page.locator(".summary-chips .pill").count();
ok("summary lists the chosen formats", sumChips === 3, `${sumChips} formats`);
ok("step 3 has the start button", (await page.locator(".upload-start").count()) > 0);
await page.screenshot({ path: `${OUT}/s3-details.png` });

// --- back / forward through the flow -----------------------------------------
await page.goBack();
await page.waitForTimeout(800);
ok("back returns to output", (await page.locator(".step-output").count()) > 0, page.url().split("#")[1] ?? "");
await page.goBack();
await page.waitForTimeout(800);
ok("back again returns to the file step", (await page.locator(".step-file").count()) > 0);
ok("the chosen file survives going back", (await page.locator(".drop.has-file").count()) > 0);
await page.goForward();
await page.waitForTimeout(800);
ok("forward works", (await page.locator(".step-output").count()) > 0);

// --- clicking a completed step on the rail -----------------------------------
await page.locator(".step-item.is-done .step-dot").first().click();
await page.waitForTimeout(800);
ok("the rail navigates to a completed step", (await page.locator(".step-file").count()) > 0);

// --- a deep link past step 1 with no file falls back -------------------------
const ctx2 = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const fresh = await ctx2.newPage();
fresh.on("pageerror", (e) => errors.push("FRESH: " + e.message));
await fresh.addInitScript((t) => localStorage.setItem("ptify.token", t), TOKEN);
await fresh.goto(`${BASE}/#/new/details`, { waitUntil: "domcontentloaded" });
await fresh.waitForFunction(() => !document.querySelector(".boot"), { timeout: 9000 }).catch(() => {});
await fresh.waitForTimeout(1500);
ok("deep link past step 1 with no file falls back",
   (await fresh.locator(".step-file").count()) > 0 && !fresh.url().includes("details"),
   fresh.url().split("#")[1] ?? "");
await ctx2.close();

console.log("\n=== UPLOAD FLOW ===");
let fails = 0;
for (const r of results) { if (!r.p) fails++; console.log(`${r.p ? "PASS" : "FAIL"}  ${r.n}${r.d ? "   [" + r.d + "]" : ""}`); }
console.log(`\n${results.length - fails}/${results.length} passed`);
console.log("console errors: " + (errors.length ? "\n  " + errors.join("\n  ") : "none"));
await browser.close();
process.exit(fails ? 1 : 0);
