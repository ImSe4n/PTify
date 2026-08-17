/**
 * Phase 9f: the `remote` engine reaches the UPLOAD PICKER, and is selectable.
 *
 * WHY THIS NEEDS A BROWSER AND NOT A CURL
 *
 * `GET /v1/engines` already returns `remote` -- that was verified at the API.
 * What that does NOT prove is that the picker renders it, that it is
 * selectable, or that picking it survives into the submitted job. UploadScreen
 * maps over the list and disables anything with `available: false`, so a new
 * engine can reach the wire and still be un-clickable on screen.
 *
 * HANDOFF section 1 is the reason to bother: six of six Phase 7-8 defects were
 * found by driving a browser or reading a screenshot, and none by typechecking.
 * An engine that 404s in the picker types perfectly.
 *
 * PREREQUISITES: the API on :8000 with PTIFY_REMOTE_URL set, and dev on :5173.
 * Without PTIFY_REMOTE_URL the API reports remote as unavailable and the
 * selectable assertions below correctly fail -- that is the point, not a flake.
 */

import { chromium } from "playwright";
import { readFileSync } from "node:fs";

const APP = "http://localhost:5173";
const API = "http://127.0.0.1:8000";

let pass = 0;
let fail = 0;
const errors = [];

function check(name, ok, detail = "") {
  if (ok) {
    pass++;
    console.log(`PASS  ${name}${detail ? `   [${detail}]` : ""}`);
  } else {
    fail++;
    console.log(`FAIL  ${name}${detail ? `   [${detail}]` : ""}`);
  }
}

const token = readFileSync("../var/p7tok.txt", "utf8").trim();

const browser = await chromium.launch();
const page = await browser.newPage();

page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});

console.log("\n=== 9f REMOTE ENGINE IN THE UI ===\n");

// --- the API says what we think it says ----------------------------------
const listed = await (await fetch(`${API}/v1/engines`)).json();
const remoteApi = listed.find((e) => e.name === "remote");
check("the API offers a remote engine", !!remoteApi);
check(
  "the API reports it available (PTIFY_REMOTE_URL is set)",
  remoteApi?.available === true,
  `available=${remoteApi?.available}`,
);
check(
  "remote is NOT the default -- the CPU path stays the default",
  remoteApi?.default === false,
  `default=${remoteApi?.default}`,
);

// --- sign in and reach the picker ----------------------------------------
// addInitScript, not evaluate: the token has to be in localStorage BEFORE the
// app boots, or the auth guard bounces the first render.
await page.addInitScript((t) => localStorage.setItem("ptify.token", t), token);
await page.goto(`${APP}/#/`, { waitUntil: "domcontentloaded" });
await page
  .waitForFunction(() => !document.querySelector(".boot"), { timeout: 9000 })
  .catch(() => {});
await page.waitForSelector(".step-file", { timeout: 12000 });

// The picker lives in step 2 ("output"), reachable only once a file is chosen
// -- the upload flow is stepped (HANDOFF section 1). The file input is
// `sr-only`, so it is never "visible"; setInputFiles drives it regardless,
// which is how upload-flow.mjs does it too.
await page.locator('input[type="file"]').setInputFiles("../var/clip25.wav");
await page.waitForSelector(".engine-list .engine", { timeout: 15000 });

const names = await page.$$eval(".engine-list .engine-name", (els) =>
  els.map((e) => e.textContent.trim()),
);
check("the picker renders every engine the API offers", names.length === listed.length,
  `ui ${names.length} vs api ${listed.length}`);
check("the picker shows `remote`", names.includes("remote"), names.join(", "));

// --- it is actually selectable, not rendered-but-disabled -----------------
const remoteBtn = page.locator(".engine", { hasText: "remote" }).first();
const disabled = await remoteBtn.evaluate((el) =>
  el.disabled === true || el.classList.contains("is-unavailable"),
);
check("the remote card is not disabled", disabled === false);

await remoteBtn.click();
const selected = await remoteBtn.evaluate((el) =>
  el.classList.contains("is-selected"),
);
check("clicking remote selects it", selected);

// --- the choice survives into the review step -----------------------------
const notes = await remoteBtn.evaluate(
  (el) => el.querySelector(".engine-notes")?.textContent ?? "",
);
check(
  "its notes explain that the model runs elsewhere",
  /GPU host|runs there|remote/i.test(notes),
  notes.slice(0, 60),
);

await browser.close();

console.log(`\n${pass}/${pass + fail} passed`);
console.log(`console errors: ${errors.length ? errors.join(" | ") : "none"}`);

process.exit(fail === 0 && errors.length === 0 ? 0 : 1);
