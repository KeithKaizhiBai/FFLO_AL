import { chromium } from "/Users/baikaizhi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const browser = await chromium.launch({
  headless: true,
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  args: ["--disable-gpu"],
});
try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(join(here, "export.html")).href, { waitUntil: "networkidle" });
  await page.evaluate(async () => {
    await document.fonts.ready;
    const image = document.querySelector("img");
    if (!image.complete) await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; });
    if (image.decode) await image.decode();
  });
  await page.emulateMedia({ media: "print" });
  await page.pdf({
    path: join(here, "fig1_engine_v3.pdf"), width: "178mm", height: "120.15mm",
    margin: { top: 0, right: 0, bottom: 0, left: 0 }, printBackground: true,
    preferCSSPageSize: true, displayHeaderFooter: false,
  });
} finally {
  await browser.close();
}
