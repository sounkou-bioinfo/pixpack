
# Rpixpack: R Bindings for PixPack

Turn **any file into a single PNG** whose **pixels carry the data** -
now from R! Feed that PNG back in to **recover the exact original** with
strong integrity checks.

This package provides R bindings for the excellent
[PixPack](https://github.com/SauersML/pixpack) utility, allowing you to
encode and decode files directly from R.

## Installation

You’ll need Rust installed on your system. Install from
[rustup.rs](https://rustup.rs/).

``` r
# Install from source (development version)
devtools::install_github("sounkou-bioinfo/pixpack", subdir = ".", ref = "Rpixpack")
```

## Quick Start

``` r
library(Rpixpack)
```

### Text Encoding Example

``` r
# Encode text to PNG
png_file <- tempfile(fileext = ".png")
result <- pixpack_text("Hello, PixPack from R! 🚀📊", png_file)
#> Encoding text to PNG...
#> Created PNG: /tmp/RtmpxVuzpM/file1c3673b22b899.png
cat("Created PNG:", result, "\n")
#> Created PNG: /tmp/RtmpxVuzpM/file1c3673b22b899.png

# Decode PNG back to text
decoded <- pixpack_text(png_path = png_file)
#> Decoding PNG to text...
#> Decoded 25 characters
cat("Decoded text:", decoded, "\n")
#> Decoded text: Hello, PixPack from R! 🚀📊

# Show file info
info <- pixpack_info(png_file)
print(info)
#> PixPack File Info:
#>   File: file1c3673b22b899.png 
#>   Size: 26.28 KB
#>   Type: Possibly PixPack PNG (use pixpack_plot for visualization) 
#>   Modified: 2025-08-09 21:44:17
```

### File Encoding Example

``` r
# Create a test file with some content
test_file <- tempfile(fileext = ".txt")
test_content <- c(
  "# Sample Data File",
  "timestamp,value,category", 
  "2024-01-01,42.5,A",
  "2024-01-02,38.1,B", 
  "2024-01-03,51.2,A"
)
writeLines(test_content, test_file)

# Encode file to PNG
png_result <- pixpack_convert(test_file, verbose = TRUE)
#> Encoding file to PNG...
#> Output: /tmp/RtmpxVuzpM/file1c36712b76133.txt.png

# Decode PNG back to original file
decoded_file <- pixpack_convert(png_result, verbose = TRUE)
#> Decoding PNG to original file...
#> Output: /tmp/RtmpxVuzpM/file1c36712b76133.txt

# Verify the content is preserved
original_content <- readLines(test_file)
decoded_content <- readLines(decoded_file)
cat("Content preserved:", identical(original_content, decoded_content), "\n")
#> Content preserved: TRUE
```

### Visualization

``` r
# Visualize the PixPack PNG structure
pixpack_plot(png_result, show_grid = TRUE)
```

<img src="inst/doc/README-plot-example-1.png" width="100%" />

    #> PixPack PNG visualized. Dimensions: 1160x1140

## Core Functions

- `pixpack_convert(file_path)`: Auto-detect encode/decode based on file
  type
- `pixpack_text(text, png_path)`: Encode text to PNG or decode PNG to
  text  
- `pixpack_plot(png_path)`: Visualize PixPack PNG structure
- `pixpack_info(file_path)`: Get file information

## Low-level Functions

- `fileConversion(path)`: Direct file encoding/decoding (auto-detect)
- `StringConversion(input, png_path)`: Direct string encoding/decoding

## How PixPack Works

- Renders a **macro-cell grid** with a **white quiet zone** and **black
  frame**.
- Packs: `MAGIC | VERSION | header (×2) | payload | trailer_u32`.
  - **Header**: filename, total length, full BLAKE3, grid hints.
  - **Trailer**: first 4 bytes of BLAKE3(payload) for a fast sanity
    check.
- **Decode**: grayscale → Otsu threshold (with a few fallbacks) → infer
  frame/quiet → sample **center pixel per cell** → rebuild stream.
- **Integrity**:
  - Encode: re-open the written PNG, decode it back, verify **length +
    full BLAKE3 + byte-for-byte**.
  - Decode: after writing the file, re-hash and verify **length + full
    BLAKE3** again.
- Capacity is **1 bit per data cell**; pixpack picks a near-square grid
  and a cell size capped by a max canvas side.

## Testing

Run the test suite:

``` r
# Run tests
tinytest::run_test_dir(system.file("tinytest", package = "Rpixpack"))
```

## Credits

- **Original PixPack**: Created by
  [SauersML](https://github.com/SauersML/pixpack)
- **R Bindings**: Implemented using the
  [savvy](https://github.com/yutannihilation/savvy) framework
- **Integration**: Built with Rust-R FFI for high performance

## License

MIT License - see LICENSE file for details.

## Repository

- **Original PixPack**: <https://github.com/SauersML/pixpack>
- **R Bindings**: <https://github.com/sounkou-bioinfo/pixpack>
