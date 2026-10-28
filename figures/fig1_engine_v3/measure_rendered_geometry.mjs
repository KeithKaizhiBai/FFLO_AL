import { chromium } from "/Users/baikaizhi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { writeFile } from "node:fs/promises";

const here = dirname(fileURLToPath(import.meta.url));
const browser = await chromium.launch({
  headless: true,
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  args: ["--disable-gpu"],
});

try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(join(here, "fig1_engine_v3.svg")).href, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  const result = await page.evaluate(() => {
    const box = (id) => {
      const node = document.getElementById(id);
      if (!node) throw new Error(`missing SVG node: ${id}`);
      const value = node.getBBox();
      return { x: value.x, y: value.y, width: value.width, height: value.height,
               right: value.x + value.width, bottom: value.y + value.height };
    };
    const inside = (outer, inner, minimum = 0) => ({
      left: inner.x - outer.x,
      top: inner.y - outer.y,
      right: outer.right - inner.right,
      bottom: outer.bottom - inner.bottom,
      pass: Math.min(inner.x - outer.x, inner.y - outer.y,
                     outer.right - inner.right, outer.bottom - inner.bottom) >= minimum,
    });
    const reliability = box("reliability-card-background");
    const reliabilityIds = ["reliability-title", "reliability-row-1", "reliability-row-2", "reliability-row-3"];
    const text = Object.fromEntries(reliabilityIds.map((id) => [id, box(id)]));
    const clearances = Object.fromEntries(reliabilityIds.map((id) => [id, inside(reliability, text[id], 4)]));
    const assetIds = ["asset-engine-v3-central", "asset-engine-v3-bdg", "asset-engine-v3-observation"];
    const assets = Object.fromEntries(assetIds.map((id) => [id, box(id)]));
    const canvas = { x: 0, y: 0, right: 1600, bottom: 1080 };
    const containment = Object.fromEntries(assetIds.map((id) => [id, inside(canvas, assets[id], 0)]));
    const failures = [
      ...Object.entries(clearances).filter(([, value]) => !value.pass).map(([id, value]) => ({ id, value })),
      ...Object.entries(containment).filter(([, value]) => !value.pass).map(([id, value]) => ({ id, value })),
    ];
    return { schema_version: 1, reliability_card: reliability, reliability_text: text, clearances,
             asset_boxes: assets, asset_containment: containment,
             status: failures.length ? "fail" : "pass", failures };
  });
  await writeFile(join(here, "rendered_geometry_audit.json"), JSON.stringify(result, null, 2) + "\n");
  console.log(JSON.stringify({ status: result.status, failures: result.failures.length }));
} finally {
  await browser.close();
}
