# NSFW Detector

A TypeScript-based tool for detecting and processing potentially inappropriate content in images using machine learning. This project uses the `nsfwjs` library with TensorFlow.js to classify images and optionally create blurred versions of flagged content.

## Features

- 🔍 **Automated NSFW Detection**: Scans images for inappropriate content using machine learning
- 📊 **Detailed Reporting**: Generates CSV reports with classification probabilities
- 🎯 **Configurable Thresholds**: Customizable sensitivity levels for different content types
- 🖼️ **Image Processing**: Optional blurring and redaction of flagged images
- ⚡ **Performance Optimized**: Uses WebAssembly backend for fast processing
- 📁 **Batch Processing**: Handles multiple images in a single run

## Classification Categories

The detector analyzes images and assigns probabilities to these categories:

- **Porn** - Explicit sexual content
- **Hentai** - Anime/manga sexual content
- **Sexy** - Suggestive but not explicit content
- **Neutral** - Safe, non-sexual content
- **Drawing** - Illustrated content

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd nsfw-detector
```

2. Install dependencies:

```bash
npm install
```

## Usage

### Basic Scanning

Scan all images in the `images` directory:

```bash
npm run scan
```

### Scan with Blurring

Scan images and create blurred versions of flagged content:

```bash
npm run scan:blur
```

### Custom Usage

You can also run the script directly with custom parameters:

```bash
npx tsx scripts/nsfw-scan.ts "path/to/images/**/*.{jpg,jpeg,png,webp}" output_directory --blur
```

**Parameters:**

- `image_glob`: Glob pattern for images to scan (default: `images/**/*.{jpg,jpeg,png,webp}`)
- `output_dir`: Directory for output files (default: `nsfw_output`)
- `--blur`: Optional flag to create blurred versions of flagged images

## Output

The tool generates:

1. **CSV Report** (`nsfw_report.csv`): Detailed analysis results including:
   - File path
   - Flagged status (true/false)
   - Top classification and probability
   - Individual category probabilities
   - Path to redacted image (if blurring enabled)

2. **Blurred Images** (optional): If `--blur` flag is used, creates heavily blurred versions of flagged images in the output directory

## Configuration

### Detection Thresholds

The tool uses conservative thresholds optimized for content moderation:

```typescript
const THRESHOLDS = {
  PORN: 0.5, // 50% confidence threshold
  HENTAI: 0.5, // 50% confidence threshold
  SEXY: 0.6, // 60% confidence threshold
};
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)

## Technical Details

- **Framework**: TypeScript with Node.js
- **ML Library**: nsfwjs v4.2.1
- **TensorFlow Backend**: WebAssembly for optimal performance
- **Image Processing**: Sharp for efficient image manipulation
- **Input Size**: Images are automatically resized to 224x224 pixels for analysis

## Project Structure

```
nsfw-detector/
├── images/                 # Input images directory
├── nsfw_output/           # Output directory for reports and blurred images
├── scripts/
│   ├── nsfw-scan.ts       # Main scanning script
│   └── wasm/              # TensorFlow WebAssembly files
├── package.json
├── tsconfig.json
└── README.md
```

## Dependencies

### Production

- `@tensorflow/tfjs`: Core TensorFlow.js library
- `@tensorflow/tfjs-backend-wasm`: WebAssembly backend for performance
- `nsfwjs`: NSFW classification model
- `sharp`: High-performance image processing
- `csv-writer`: CSV file generation
- `fast-glob`: File pattern matching

### Development

- `typescript`: TypeScript compiler
- `tsx`: TypeScript execution
- `@types/node`: Node.js type definitions

## Performance Notes

- Uses WebAssembly backend for optimal performance
- Images are resized before tensor conversion to minimize memory usage
- Tensor memory is properly disposed after each classification
- Batch processing for efficient handling of large image collections

## License

ISC

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Disclaimer

This tool is designed for content moderation purposes. The accuracy of NSFW detection may vary, and manual review is recommended for critical applications. Use responsibly and in accordance with applicable laws and regulations.
