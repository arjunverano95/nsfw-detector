import fs from 'node:fs';
import path from 'node:path';
import fg from 'fast-glob';
import sharp from 'sharp';
// Import the high‑level TFJS API.  In recent versions of tfjs you should
// explicitly enable production mode before loading any models.  This
// disables expensive runtime checks and increases inference throughput【382450127579223†L410-L515】.
import * as tf from '@tensorflow/tfjs';
import {setWasmPaths} from '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-wasm';
// nsfwjs is published as a CommonJS module.  When targeting an ESM
// environment, use createRequire to access it.  Newer versions of
// nsfwjs (>= 4.2.x) continue to provide the same API surface【382450127579223†L410-L515】.
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
const nsfw: typeof import('nsfwjs') = require('nsfwjs');
// The csv‑writer package is still published as CommonJS.  It does not
// provide a default export when esModuleInterop is disabled, so import
// the named factory directly【87673405033056†L33-L49】.
import {createObjectCsvWriter} from 'csv-writer';

/**
 * Structure returned by nsfwjs.classify().  Each prediction contains a
 * className and a probability.
 */
type Pred = {className: string; probability: number};

// Patterns, directories and flags are read from process.argv.  The code
// defaults to scanning all common raster image types under the
// `images` folder and will write the report and optionally blurred
// images into the `nsfw_output` folder.  The flag `--blur` causes
// flagged images to be strongly blurred and saved alongside the report.
const IMG_GLOB = process.argv[2] || 'images/**/*.{jpg,jpeg,png,webp}';
const OUT_DIR = process.argv[3] || 'nsfw_output';
const BLUR_FLAGGED = process.argv.includes('--blur');
// Number of top predictions to record.  nsfwjs defaults to 5 classes.
const TOPK = 5;
// Determine where the wasm backend will look for the compiled binaries.
// In modern versions of @tensorflow/tfjs-backend-wasm you should call
// setWasmPaths() before switching backends【855699054060637†L112-L129】.
const wasmDir =
  path.resolve('node_modules/@tensorflow/tfjs-backend-wasm/wasm-out') +
  path.sep;

// Tune classification thresholds.  Apple’s App Store review is
// particularly conservative, so we err on the side of caution.
const THRESHOLDS = {
  PORN: 0.1,
  HENTAI: 0.15,
  SEXY: 0.3,
};

/**
 * Determine whether a given set of predictions should be considered
 * objectionable.  Any category above its threshold will result in a
 * “flagged” verdict.
 */
function isFlagged(preds: Pred[]): boolean {
  const get = (name: string) =>
    preds.find((p) => p.className.toLowerCase() === name.toLowerCase())
      ?.probability || 0;
  const porn = get('Porn');
  const hentai = get('Hentai');
  const sexy = get('Sexy');
  return (
    porn >= THRESHOLDS.PORN ||
    hentai >= THRESHOLDS.HENTAI ||
    sexy >= THRESHOLDS.SEXY
  );
}

/**
 * Ensure a directory exists on disk.  Uses fs.promises.mkdir with the
 * recursive option enabled.
 */
async function ensureDir(p: string): Promise<void> {
  await fs.promises.mkdir(p, {recursive: true});
}

/**
 * Convert an image file into a Tensor.  The WASM backend requires
 * fixed‑size inputs.  Resize the image down to 224×224 pixels using
 * sharp before constructing a tensor.  Leaving the data as Int32Array
 * mirrors the behaviour of tf.browser.fromPixels()【382450127579223†L602-L611】.
 */
async function loadImageAsTensor(fp: string): Promise<tf.Tensor3D> {
  const {data, info} = await sharp(fp)
    .removeAlpha()
    .resize(224, 224, {fit: 'cover'})
    .raw()
    .toBuffer({resolveWithObject: true});
  return tf.tensor3d(data, [info.height, info.width, 3], 'int32');
}

/**
 * Create a heavily blurred copy of an image.  Uses sharp.blur() with
 * a high sigma for a strong Gaussian blur【962679667679834†L477-L513】.  The output
 * directory structure mirrors the input’s relative path.
 */
async function blurImage(inputPath: string, outputPath: string): Promise<void> {
  await ensureDir(path.dirname(outputPath));
  const img = sharp(inputPath);
  const {width, height} = await img.metadata();
  if (!width || !height) {
    await img.blur(20).toFile(outputPath);
    return;
  }
  await img.blur(20).toFile(outputPath);
}

async function main(): Promise<void> {
  // Enable production mode to disable costly runtime checks【382450127579223†L506-L515】.
  tf.enableProdMode();
  // Configure the WASM backend.  If you omit setWasmPaths() then TFJS will
  // attempt to locate the wasm binaries relative to the compiled bundle.
  setWasmPaths(wasmDir);
  await tf.setBackend('wasm');
  await tf.ready();

  console.log(`Scanning: ${IMG_GLOB}`);
  const files = await fg(IMG_GLOB, {dot: false, onlyFiles: true});
  if (files.length === 0) {
    console.log('No images found.');
    return;
  }
  await ensureDir(OUT_DIR);
  // Load the default MobileNetV2 model.  The latest nsfwjs library still
  // supports this API【382450127579223†L410-L515】.
  const model = await nsfw.load('MobileNetV2Mid'); // MobileNetV2, MobileNetV2Mid, InceptionV3
  const rows: any[] = [];
  for (const fp of files) {
    try {
      const tensor = await loadImageAsTensor(fp);
      // Pass TOPK to limit the number of returned predictions【382450127579223†L486-L503】.
      const preds = (await model.classify(tensor, TOPK)) as Pred[];
      tensor.dispose();
      preds.sort((a, b) => b.probability - a.probability);
      const flagged = isFlagged(preds);
      let redactedPath = '';
      if (flagged && BLUR_FLAGGED) {
        const rel = path.relative(process.cwd(), fp);
        const outPath = path.join(
          OUT_DIR,
          rel.replace(path.extname(rel), '') + '.blurred.jpg',
        );
        await blurImage(fp, outPath);
        redactedPath = outPath;
      }
      rows.push({
        file: fp,
        flagged,
        top1_label: preds[0]?.className || '',
        top1_prob: preds[0]?.probability.toFixed(4) || '',
        porn:
          preds.find((p) => p.className === 'Porn')?.probability.toFixed(4) ||
          '0',
        hentai:
          preds.find((p) => p.className === 'Hentai')?.probability.toFixed(4) ||
          '0',
        sexy:
          preds.find((p) => p.className === 'Sexy')?.probability.toFixed(4) ||
          '0',
        neutral:
          preds
            .find((p) => p.className === 'Neutral')
            ?.probability.toFixed(4) || '0',
        drawing:
          preds
            .find((p) => p.className === 'Drawing')
            ?.probability.toFixed(4) || '0',
        redacted_path: redactedPath,
      });
      console.log(
        `${flagged ? '🚫' : '✅'} ${fp}  ` +
          preds
            .map((p) => `${p.className}:${p.probability.toFixed(2)}`)
            .join(' | '),
      );
    } catch (e) {
      console.error(`Error processing ${fp}:`, e);
    }
  }
  const csvWriter = createObjectCsvWriter({
    path: path.join(OUT_DIR, 'nsfw_report.csv'),
    header: [
      {id: 'file', title: 'file'},
      {id: 'flagged', title: 'flagged'},
      {id: 'top1_label', title: 'top1_label'},
      {id: 'top1_prob', title: 'top1_prob'},
      {id: 'porn', title: 'porn'},
      {id: 'hentai', title: 'hentai'},
      {id: 'sexy', title: 'sexy'},
      {id: 'neutral', title: 'neutral'},
      {id: 'drawing', title: 'drawing'},
      {id: 'redacted_path', title: 'redacted_path'},
    ],
    fieldDelimiter: ',',
    alwaysQuote: true,
  });
  await csvWriter.writeRecords(rows);
  console.log(`\nReport: ${path.join(OUT_DIR, 'nsfw_report.csv')}`);
  if (BLUR_FLAGGED) console.log(`Blurred copies saved under: ${OUT_DIR}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
