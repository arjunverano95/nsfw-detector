import fs from "node:fs";
import path from "node:path";
import fg from "fast-glob";
import sharp from "sharp";
import * as tf from "@tensorflow/tfjs";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import "@tensorflow/tfjs-backend-wasm";
// nsfwjs has no default export in ESM:
// import * as nsfw from "nsfwjs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const nsfw: typeof import("nsfwjs") = require("nsfwjs");
// csv-writer is CJS; named import works with esModuleInterop=false/true under NodeNext:
import { createObjectCsvWriter } from "csv-writer";

type Pred = { className: string; probability: number };

const IMG_GLOB = process.argv[2] || "images/**/*.{jpg,jpeg,png,webp}";
const OUT_DIR = process.argv[3] || "nsfw_output";
const BLUR_FLAGGED = process.argv.includes("--blur");
const TOPK = 5;
const wasmDir =
  path.resolve("node_modules/@tensorflow/tfjs-backend-wasm/wasm-out") +
  path.sep;
// Thresholds (tune for Apple’s conservative review)
const THRESHOLDS = {
  PORN: 0.4,
  HENTAI: 0.4,
  SEXY: 0.6, // “Sexy” can still be objectionable for 1.1; keep strict
};

const isFlagged = (preds: Pred[]) => {
  const get = (name: string) =>
    preds.find((p) => p.className.toLowerCase() === name.toLowerCase())
      ?.probability || 0;

  const porn = get("Porn");
  const hentai = get("Hentai");
  const sexy = get("Sexy");

  // Any explicit-ish score beyond thresholds flags the image
  return (
    porn >= THRESHOLDS.PORN ||
    hentai >= THRESHOLDS.HENTAI ||
    sexy >= THRESHOLDS.SEXY
  );
};

async function ensureDir(p: string) {
  await fs.promises.mkdir(p, { recursive: true });
}

async function loadImageAsTensor(fp: string) {
  // ✅ Resize BEFORE converting to a tensor to keep WASM memory tiny
  const { data, info } = await sharp(fp)
    .removeAlpha()
    .resize(224, 224, { fit: "cover" }) // nsfwjs default input size
    .raw()
    .toBuffer({ resolveWithObject: true });

  // data is Uint8Array length 224*224*3
  // nsfwjs works with int32 or float; int32 mirrors fromPixels
  return tf.tensor3d(data, [info.height, info.width, 3], "int32");
}

async function blurImage(inputPath: string, outputPath: string) {
  await ensureDir(path.dirname(outputPath));
  // strong blur; also add black bar as fallback for predictable redaction
  const img = sharp(inputPath);
  const { width, height } = await img.metadata();

  if (!width || !height) {
    await img.blur(20).toFile(outputPath);
    return;
  }

  // heavy blur pass
  const blurred = await img.blur(20).toBuffer();

  // draw a black overlay at center (belt-and-suspenders)
  const overlay = Buffer.from(
    `<svg width="${width}" height="${height}">
      <rect x="0" y="${Math.floor(
        height * 0.3
      )}" width="${width}" height="${Math.floor(
      height * 0.4
    )}" fill="black" opacity="0.85"/>
    </svg>`
  );

  await sharp(blurred)
    .composite([{ input: overlay, top: 0, left: 0 }])
    .toFile(outputPath);
}

async function main() {
  // setWasmPaths(
  //   "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/"
  // );
  setWasmPaths(wasmDir);
  await tf.setBackend("wasm");
  await tf.ready();

  console.log(`Scanning: ${IMG_GLOB}`);
  const files = await fg(IMG_GLOB, { dot: false, onlyFiles: true });
  if (files.length === 0) {
    console.log("No images found.");
    return;
  }

  await ensureDir(OUT_DIR);

  const model = await nsfw.load(); // loads the default lightweight model
  const rows: any[] = [];

  for (const fp of files) {
    try {
      const tensor = await loadImageAsTensor(fp);
      const preds = (await model.classify(tensor, TOPK)) as Pred[];
      tensor.dispose();

      // sort by probability desc
      preds.sort((a, b) => b.probability - a.probability);

      const flagged = isFlagged(preds);

      // optional blur output
      let redactedPath = "";
      if (flagged && BLUR_FLAGGED) {
        const rel = path.relative(process.cwd(), fp);
        const outPath = path.join(
          OUT_DIR,
          rel.replace(path.extname(rel), "") + ".blurred.jpg"
        );
        await blurImage(fp, outPath);
        redactedPath = outPath;
      }

      rows.push({
        file: fp,
        flagged,
        top1_label: preds[0]?.className || "",
        top1_prob: preds[0]?.probability.toFixed(4) || "",
        porn:
          preds.find((p) => p.className === "Porn")?.probability.toFixed(4) ||
          "0",
        hentai:
          preds.find((p) => p.className === "Hentai")?.probability.toFixed(4) ||
          "0",
        sexy:
          preds.find((p) => p.className === "Sexy")?.probability.toFixed(4) ||
          "0",
        neutral:
          preds
            .find((p) => p.className === "Neutral")
            ?.probability.toFixed(4) || "0",
        drawing:
          preds
            .find((p) => p.className === "Drawing")
            ?.probability.toFixed(4) || "0",
        redacted_path: redactedPath,
      });

      console.log(
        `${flagged ? "🚫" : "✅"} ${fp}  ` +
          preds
            .map((p) => `${p.className}:${p.probability.toFixed(2)}`)
            .join(" | ")
      );
    } catch (e) {
      console.error(`Error processing ${fp}:`, e);
    }
  }

  const csvWriter = createObjectCsvWriter({
    path: path.join(OUT_DIR, "nsfw_report.csv"),
    header: [
      { id: "file", title: "file" },
      { id: "flagged", title: "flagged" },
      { id: "top1_label", title: "top1_label" },
      { id: "top1_prob", title: "top1_prob" },
      { id: "porn", title: "porn" },
      { id: "hentai", title: "hentai" },
      { id: "sexy", title: "sexy" },
      { id: "neutral", title: "neutral" },
      { id: "drawing", title: "drawing" },
      { id: "redacted_path", title: "redacted_path" },
    ],
    fieldDelimiter: ",",
    alwaysQuote: true,
  });

  await csvWriter.writeRecords(rows);
  console.log(`\nReport: ${path.join(OUT_DIR, "nsfw_report.csv")}`);
  if (BLUR_FLAGGED) console.log(`Blurred copies saved under: ${OUT_DIR}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
