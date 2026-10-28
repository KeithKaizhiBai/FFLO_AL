import { chromium } from "/Users/baikaizhi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const browser = await chromium.launch({
  headless: true,
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  args: ["--disable-gpu"],
});

async function render(relative, output, width, height) {
  const page = await browser.newPage({ viewport: { width, height } });
  await page.goto(pathToFileURL(join(here, relative)).href, { waitUntil: "load" });
  await page.evaluate(async ({ width, height }) => {
    const svg = document.documentElement;
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    svg.style.width = `${width}px`;
    svg.style.height = `${height}px`;
    svg.style.display = "block";
    await document.fonts.ready;
  }, { width, height });
  await page.screenshot({ path: join(here, output), omitBackground: false });
  await page.close();
}

try {
  await render("fig1_engine_v4.svg", "fig1_engine_v4_svg_preview.png", 1600, 1080);
  await render("asset_versions/thermodynamic_landscape_v4/asset.svg", "central_panel_preview.png", 925, 690);
} finally {
  await browser.close();
}
