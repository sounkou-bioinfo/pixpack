# Rpixpack: R Bindings for PixPack

Turn **any file into a single PNG** whose **pixels carry the data** - now from R! Feed that PNG back in to **recover the exact original** with strong integrity checks.

This package provides R bindings for the excellent [PixPack](https://github.com/SauersML/pixpack) utility, allowing you to encode and decode files directly from R.

## Installation

You'll need Rust installed on your system. Install from [rustup.rs](https://rustup.rs/).

```r
# Install from source (development version)
devtools::install_github("sounkou-bioinfo/pixpack", subdir = ".", ref = "Rpixpack")
```

## Quick Start

```r
library(Rpixpack)

# Encode text to PNG
png_file <- tempfile(fileext = ".png")
result <- pixpack_text("Hello, PixPack from R!", png_file)

# Decode PNG back to text
decoded <- pixpack_text(png_path = png_file)
print(decoded)  # "Hello, PixPack from R!"

# Encode any file to PNG
test_file <- tempfile(fileext = ".txt")
writeLines(c("Line 1", "Line 2", "Line 3"), test_file)
png_result <- pixpack_convert(test_file)

# Decode PNG back to original file
decoded_file <- pixpack_convert(png_result)

# Visualize a PixPack PNG
pixpack_plot(png_result)

# Get file information
pixpack_info(test_file)
```

## Core Functions

- `pixpack_convert(file_path)`: Auto-detect encode/decode based on file type
- `pixpack_text(text, png_path)`: Encode text to PNG or decode PNG to text  
- `pixpack_plot(png_path)`: Visualize PixPack PNG structure
- `pixpack_info(file_path)`: Get file information

## Low-level Functions

- `fileConversion(path)`: Direct file encoding/decoding (auto-detect)
- `StringConversion(input, png_path)`: Direct string encoding/decoding

## How PixPack Works

- Renders a **macro-cell grid** with a **white quiet zone** and **black frame**.
- Packs: `MAGIC | VERSION | header (×2) | payload | trailer_u32`.
  - **Header**: filename, total length, full BLAKE3, grid hints.
  - **Trailer**: first 4 bytes of BLAKE3(payload) for a fast sanity check.
- **Decode**: grayscale → Otsu threshold (with a few fallbacks) → infer frame/quiet → sample **center pixel per cell** → rebuild stream.
- **Integrity**:
  - Encode: re-open the written PNG, decode it back, verify **length + full BLAKE3 + byte-for-byte**.
  - Decode: after writing the file, re-hash and verify **length + full BLAKE3** again.
- Capacity is **1 bit per data cell**; pixpack picks a near-square grid and a cell size capped by a max canvas side.

## Testing

Run the test suite:

```r
# Run tests
tinytest::run_test_dir(system.file("tinytest", package = "Rpixpack"))
```

## Credits

- **Original PixPack**: Created by [SauersML](https://github.com/SauersML/pixpack)
- **R Bindings**: Implemented using the [savvy](https://github.com/yutannihilation/savvy) framework
- **Integration**: Built with Rust-R FFI for high performance

## License

MIT License - see LICENSE file for details.

## Repository

- **Original PixPack**: <https://github.com/SauersML/pixpack>
- **R Bindings**: <https://github.com/sounkou-bioinfo/pixpack>
