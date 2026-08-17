import { chromium } from "playwright";
import fs from "node:fs";

const TOKEN = fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7tok.txt", "utf8").trim();
const JOB = JSON.parse(fs.readFileSync("C:/Users/SeanN/LivePianoSynthesizer/var/p7job.json", "utf8")).job_id;
const BASE = "http://localhost:5173";

const results = [];
const ok = (name, pass, detail = "") =>
  results.push({ name, pass, detail });

const browser = await chromium.launch();
const ctx = await browser.newContext();
const page = await ctx.newPage();

const consoleErrors = [];
page.on("console", (m) => {
  if (m.type() === "error" && !m.text().includes("404")) consoleErrors.push(m.text());
});
page.on("pageerror", (e) => consoleErrors.push("PAGEERROR: " + e.message));

// Seed the token so we are signed in.
await page.addInitScript((t) => {
  localStorage.setItem("ptify.token", t);
}, TOKEN);

async function go(hash) {
  await page.goto(BASE + hash, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(900);
}

// --- 1. root renders upload -------------------------------------------------
await go("/#/");
ok("root -> upload", await page.locator(".upload").count() > 0,
   "h1=" + (await page.locator("h1").first().textContent().catch(() => "?")));

// --- 2. garbage hash falls back to upload -----------------------------------
await go("/#/garbage/nonsense?x=1");
ok("#/garbage -> upload (no blank screen)", await page.locator(".upload").count() > 0);

// --- 3. history route -------------------------------------------------------
await go("/#/history");
ok("#/history -> history", await page.locator(".history").count() > 0);

// --- 4. deep link into the job ---------------------------------------------
await go("/#/j/" + JOB);
await page.waitForSelector(".result, .waiting", { timeout: 8000 }).catch(() => {});
const isWaiting = await page.locator(".waiting").count() > 0;
const isResult = await page.locator(".result").count() > 0;
ok("#/j/{id} -> waiting OR result (state decides, not URL)", isWaiting || isResult,
   isWaiting ? "waiting" : isResult ? "result" : "NEITHER");

// --- 5. the URL survives a refresh ------------------------------------------
await page.reload({ waitUntil: "domcontentloaded" });
await page.waitForTimeout(900);
ok("refresh keeps the job URL", page.url().includes("/j/" + JOB), page.url());

// --- 6. unknown job id shows an error, not a blank -------------------------
await go("/#/j/deadbeefdeadbeefdeadbeefdeadbeef");
const notFound = await page.locator(".page").count() > 0;
const bodyText = (await page.locator("body").innerText()).slice(0, 120).replace(/\n/g, " ");
ok("unknown job -> error screen, not blank", notFound && bodyText.length > 10, bodyText);

// --- 7. back / forward ------------------------------------------------------
await go("/#/");
await go("/#/history");
await page.goBack();
await page.waitForTimeout(700);
const backOk = await page.locator(".upload").count() > 0;
await page.goForward();
await page.waitForTimeout(700);
const fwdOk = await page.locator(".history").count() > 0;
ok("back -> upload", backOk);
ok("forward -> history", fwdOk);

// --- 8. header nav writes the hash -----------------------------------------
await go("/#/");
await page.locator(".nav-link", { hasText: "Transcriptions" }).click();
await page.waitForTimeout(700);
ok("nav click -> #/history", page.url().includes("#/history"), page.url());

// --- 9. nav active state tracks the route ----------------------------------
const activeText = await page.locator(".nav-link.is-active").first().textContent().catch(() => null);
ok("active nav link is Transcriptions", activeText === "Transcriptions", String(activeText));

// --- 10. signed OUT deep link keeps the hash -------------------------------
const ctx2 = await browser.newContext();
const anon = await ctx2.newPage();
anon.on("pageerror", (e) => consoleErrors.push("ANON PAGEERROR: " + e.message));
await anon.goto(BASE + "/#/j/" + JOB, { waitUntil: "domcontentloaded" });
await anon.waitForTimeout(1200);
const onAuth = await anon.locator(".auth").count() > 0;
ok("signed-out deep link -> auth screen", onAuth);
ok("signed-out deep link PRESERVES the hash", anon.url().includes("/j/" + JOB), anon.url());
await ctx2.close();

console.log("\n=== PHASE 7a ===");
let fails = 0;
for (const r of results) {
  if (!r.pass) fails++;
  console.log(`${r.pass ? "PASS" : "FAIL"}  ${r.name}${r.detail ? "   [" + r.detail + "]" : ""}`);
}
console.log(`\n${results.length - fails}/${results.length} passed`);
console.log("console errors: " + (consoleErrors.length === 0 ? "none" : "\n  " + consoleErrors.join("\n  ")));

await browser.close();
process.exit(fails > 0 || consoleErrors.length > 0 ? 1 : 0);
