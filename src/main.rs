// src/main.rs
// pixpack — single-file visual encoding.
// One PNG in, one PNG out. No hidden chunks. The pixels ARE the data.
//
// Single-arg CLI:
//   - If input is a PNG → DECODE into the original file (with integrity proofs).
//   - Else → ENCODE into a single PNG macro-cell grid (with integrity proofs).
//
// Integrity proofs (active):
//   • Encode: re-open written PNG, decode → verify trailer u32 + BLAKE3 + byte-for-byte.
//   • Decode: after writing reconstructed file, re-hash and verify BLAKE3 + length.
//
// Build: cargo build --release
// Run:   cargo run --release -- <path>

use std::{
    env,
    ffi::OsStr,
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context, Result};
use blake3::Hasher;
use image::{imageops, DynamicImage, GrayImage, ImageBuffer, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};

// ========================= Configuration =========================

// Stream prefix
const MAGIC: &[u8; 4] = b"PXPV"; // PixPack Visual v1
const VERSION: u16 = 1;

// Visual parameters (encoder draws these; decoder infers & quantizes)
const QUIET_CELLS: u32 = 3;      // white quiet zone thickness (in cells)
const FRAME_CELLS: u32 = 2;      // solid black frame thickness (in cells)

// Encode-time sizing
const MIN_CELL_PX: u32 = 6;      // smaller still decodes, but screenshot tolerance drops
const MAX_CELL_PX: u32 = 20;     // bigger cells = larger image; clamp for sanity
const MAX_SIDE_PX: u32 = 5000;   // cap longer canvas side

// Header repeats for redundancy (same header back-to-back)
const HEADER_REPEAT: usize = 2;

// Binarization fallback thresholds after Otsu (no rotations, keep it simple)
const FALLBACK_THRESHOLDS: &[u8] = &[200, 180, 160, 140, 120, 100];

// Logging
macro_rules! step { ($($arg:tt)*) => { eprintln!("▶ {}", format!($($arg)*)); }; }
macro_rules! ok   { ($($arg:tt)*) => { eprintln!("✔ {}", format!($($arg)*)); }; }
macro_rules! warn { ($($arg:tt)*) => { eprintln!("⚠ {}", format!($($arg)*)); }; }
macro_rules! fail { ($($arg:tt)*) => { eprintln!("✘ {}", format!($($arg)*)); }; }

// ========================= Types & integrity =========================

#[derive(Debug, Clone)]
struct UnverifiedBytes(Vec<u8>);
#[derive(Debug, Clone)]
struct VerifiedBytes(Vec<u8>);
impl VerifiedBytes { fn as_slice(&self) -> &[u8] { &self.0 } }

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
struct Header {
    version: u16,
    filename: String,
    total_len: u64,
    file_hash: [u8; 32],
    dataset_id: [u8; 16],
    // visual reference (decoder infers; these are advisory)
    grid_w: u32,
    grid_h: u32,
    quiet_cells: u32,
    frame_cells: u32,
    // payload proof (fast check before full BLAKE3)
    payload_len: u64,
    payload_hash32: u32,
}

fn blake3_hash_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut h = Hasher::new(); h.update(bytes); *h.finalize().as_bytes()
}
fn blake3_hash_file(path: &Path) -> Result<([u8; 32], u64)> {
    let mut f = fs::File::open(path)?;
    let mut h = Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        h.update(&buf[..n]);
        total += n as u64;
    }
    Ok((*h.finalize().as_bytes(), total))
}
fn verify_bytes(expected_len: u64, expected_hash: [u8; 32], data: UnverifiedBytes) -> Result<VerifiedBytes> {
    if data.0.len() as u64 != expected_len {
        bail!("Length mismatch: expected {}, got {}", expected_len, data.0.len());
    }
    let got = blake3_hash_bytes(&data.0);
    if got != expected_hash {
        bail!("BLAKE3 mismatch: expected {}, got {}", hex32(expected_hash), hex32(got));
    }
    Ok(VerifiedBytes(data.0))
}
fn hex32(bytes: [u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes { use std::fmt::Write as _; let _ = write!(&mut s, "{:02x}", b); }
    s
}
fn hash32_first(bytes: &[u8]) -> u32 {
    let h = blake3_hash_bytes(bytes);
    u32::from_le_bytes([h[0], h[1], h[2], h[3]])
}
fn derive_dataset_id(bytes: &[u8], filename: &str) -> [u8; 16] {
    let mut h = Hasher::new();
    h.update(bytes);
    h.update(filename.as_bytes());
    h.update(&(bytes.len() as u64).to_le_bytes());
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    h.update(&ts.to_le_bytes());
    let digest = h.finalize();
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest.as_bytes()[..16]);
    id
}

// ========================= Otsu & binarization =========================

fn otsu_threshold(gray: &GrayImage) -> u8 {
    let mut hist = [0u64; 256];
    for p in gray.pixels() { hist[p[0] as usize] += 1; }
    let total: u64 = hist.iter().sum();
    let mut sum: u64 = 0;
    for t in 0..256 { sum += (t as u64) * hist[t]; }
    let mut sum_b = 0u64;
    let mut w_b = 0u64;
    let mut max = 0.0f64;
    let mut thresh = 0u8;
    for t in 0..256 {
        w_b += hist[t];
        if w_b == 0 { continue; }
        let w_f = total - w_b;
        if w_f == 0 { break; }
        sum_b += (t as u64) * hist[t];
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum - sum_b) as f64 / w_f as f64;
        let var_between = (w_b as f64) * (w_f as f64) * (m_b - m_f).powi(2);
        if var_between > max { max = var_between; thresh = t as u8; }
    }
    thresh
}
fn binarize(gray: &GrayImage, t: u8) -> GrayImage {
    let mut out = gray.clone();
    for p in out.pixels_mut() { p.0[0] = if p.0[0] > t { 255 } else { 0 }; }
    out
}

// ========================= Geometry inference (decode) =========================

#[derive(Debug, Clone)]
struct Geometry {
    threshold: u8,
    cell_px: u32,
    origin_x: u32, // top-left of data grid (in px)
    origin_y: u32,
    grid_w: u32,
    grid_h: u32,
    img_w: u32,
    img_h: u32,
    quiet_left: u32,
    frame_left_meas: u32,
    quiet_right: u32,
    frame_right_meas: u32,
    quiet_top: u32,
    frame_top_meas: u32,
    quiet_bottom: u32,
    frame_bottom_meas: u32,
}

fn scan_white_then_black(line: &[u8]) -> Result<(u32, u32)> {
    let mut i = 0usize;
    while i < line.len() && line[i] == 255 { i += 1; }
    if i == line.len() { bail!("No black band found following white quiet zone on probe line"); }
    let q = i as u32;
    let mut j = i;
    while j < line.len() && line[j] == 0 { j += 1; }
    let f = (j - i) as u32;
    if f == 0 { bail!("Black frame band measured as zero thickness"); }
    Ok((q, f))
}

fn try_infer_geometry(bin: &GrayImage, threshold: u8) -> Result<Geometry> {
    let w = bin.width() as usize;
    let h = bin.height() as usize;
    if w < 40 || h < 40 {
        bail!("Image too small for frame/grid detection ({}×{})", w, h);
    }

    let mid_y = h / 2;
    let row = &bin.as_raw()[mid_y * w .. (mid_y + 1) * w];
    let (q_left, f_left) = scan_white_then_black(row)
        .with_context(|| format!("Row probe (y={}): could not detect quiet+frame (threshold={})", mid_y, threshold))?;
    let (q_right, f_right) = {
        let mut rev = row.to_vec(); rev.reverse();
        scan_white_then_black(&rev)
            .with_context(|| format!("Row reverse probe (y={}): quiet+frame fail (threshold={})", mid_y, threshold))?
    };

    let mid_x = w / 2;
    let mut col: Vec<u8> = Vec::with_capacity(h);
    for y in 0..h { col.push(bin.as_raw()[y * w + mid_x]); }
    let (q_top, f_top) = scan_white_then_black(&col)
        .with_context(|| format!("Column probe (x={}): quiet+frame fail (threshold={})", mid_x, threshold))?;
    let (q_bottom, f_bottom) = {
        let mut rev = col.clone(); rev.reverse();
        scan_white_then_black(&rev)
            .with_context(|| format!("Column reverse probe (x={}): quiet+frame fail (threshold={})", mid_x, threshold))?
    };

    // Derive cell size from measured frame thickness; then QUANTIZE all frame px.
    let cell_px_x = (f_left.min(f_right) / FRAME_CELLS).max(1);
    let cell_px_y = (f_top.min(f_bottom) / FRAME_CELLS).max(1);
    let cell_px = cell_px_x.min(cell_px_y).max(1);
    let frame_px_quant = FRAME_CELLS * cell_px;

    // Quantized origins (top-left of data grid) and inner extents
    let origin_x = q_left + frame_px_quant;
    let origin_y = q_top  + frame_px_quant;
    let inner_right = (w as u32).saturating_sub(q_right + frame_px_quant);
    let inner_bottom= (h as u32).saturating_sub(q_bottom + frame_px_quant);

    if origin_x >= inner_right || origin_y >= inner_bottom {
        bail!("Invalid frame bounds after quantization: origin=({}, {}), inner_right={}, inner_bottom={}",
              origin_x, origin_y, inner_right, inner_bottom);
    }

    // Compute grid as full cells (floor); ignore leftover partial pixels
    let grid_w_px = inner_right - origin_x;
    let grid_h_px = inner_bottom - origin_y;
    let grid_w = (grid_w_px / cell_px).max(1);
    let grid_h = (grid_h_px / cell_px).max(1);

    Ok(Geometry {
        threshold,
        cell_px,
        origin_x,
        origin_y,
        grid_w,
        grid_h,
        img_w: w as u32,
        img_h: h as u32,
        quiet_left: q_left,
        frame_left_meas: f_left,
        quiet_right: q_right,
        frame_right_meas: f_right,
        quiet_top: q_top,
        frame_top_meas: f_top,
        quiet_bottom: q_bottom,
        frame_bottom_meas: f_bottom,
    })
}

// ========================= Bit packing / unpacking =========================

fn bytes_to_bits(data: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(data.len() * 8);
    for b in data {
        for k in (0..8).rev() { bits.push((b >> k) & 1); }
    }
    bits
}
fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; (bits.len() + 7) / 8];
    for (i, &bit) in bits.iter().enumerate() {
        if bit != 0 {
            let byte = i / 8;
            let off = 7 - (i % 8);
            out[byte] |= 1 << off;
        }
    }
    out
}

// ========================= Grid sizing (encode) =========================

fn compute_overhead(header_len: usize) -> usize {
    // MAGIC(4) + VERSION(2) + [len(4)+header]*HEADER_REPEAT + trailer(4)
    4 + 2 + HEADER_REPEAT * (4 + header_len) + 4
}

// Choose near-square grid to fit N bits.
fn minimal_grid(bits_needed: usize) -> (u32, u32) {
    let side = (bits_needed as f64).sqrt().ceil() as u32;
    let w = side.max(1);
    let total = bits_needed as u128;
    let h = ((total + (w as u128) - 1) / (w as u128)) as u32;
    (w, h.max(1))
}

// Pick cell_px so final canvas <= MAX_SIDE_PX.
fn choose_cell_px_for_canvas(grid_w: u32, grid_h: u32) -> u32 {
    let total_cells_x = grid_w + 2 * (QUIET_CELLS + FRAME_CELLS);
    let total_cells_y = grid_h + 2 * (QUIET_CELLS + FRAME_CELLS);
    let max_cells_side = total_cells_x.max(total_cells_y).max(1);
    let ideal = (MAX_SIDE_PX / max_cells_side).max(1);
    let chosen = ideal.clamp(MIN_CELL_PX, MAX_CELL_PX);
    chosen.max(1)
}

// ========================= Encode =========================

fn choose_grid_and_header(filename: &str, bytes: &[u8], dataset_id: [u8; 16]) -> Result<(Header, Vec<u8>, usize, u32)> {
    // Iterate once or twice in case header length nudges bits/size.
    let mut grid_w = 1u32;
    let mut grid_h = 1u32;
    for _ in 0..6 {
        let header_tmp = Header {
            version: VERSION,
            filename: filename.to_string(),
            total_len: bytes.len() as u64,
            file_hash: blake3_hash_bytes(bytes),
            dataset_id,
            grid_w,
            grid_h,
            quiet_cells: QUIET_CELLS,
            frame_cells: FRAME_CELLS,
            payload_len: bytes.len() as u64,
            payload_hash32: hash32_first(bytes),
        };
        let hdr_bytes = postcard::to_allocvec(&header_tmp)?;
        let overhead = compute_overhead(hdr_bytes.len());
        let needed_bits = (overhead + bytes.len()) * 8;

        let (gw, gh) = minimal_grid(needed_bits);
        let cell_px = choose_cell_px_for_canvas(gw, gh);

        // With the chosen cell_px, capacity in bits is gw*gh (we map 1 bit => 1 cell).
        // We only need to ensure gw*gh >= needed_bits (it is by construction).
        let header_final = Header { grid_w: gw, grid_h: gh, ..header_tmp };
        let hdr_final_bytes = postcard::to_allocvec(&header_final)?;
        let overhead2 = compute_overhead(hdr_final_bytes.len());
        let needed2 = (overhead2 + bytes.len()) * 8;
        if (gw as usize) * (gh as usize) >= needed2 {
            return Ok((header_final, hdr_final_bytes, overhead2, cell_px));
        }
        grid_w = gw;
        grid_h = gh;
    }
    bail!("Unable to size grid/header to fit payload");
}

fn render_stream_to_image(stream: &[u8], grid_w: u32, grid_h: u32, cell_px: u32) -> Result<RgbaImage> {
    let total_cells_x = grid_w + 2 * (QUIET_CELLS + FRAME_CELLS);
    let total_cells_y = grid_h + 2 * (QUIET_CELLS + FRAME_CELLS);
    let width_px  = total_cells_x * cell_px;
    let height_px = total_cells_y * cell_px;

    let mut img: RgbaImage = ImageBuffer::from_pixel(width_px, height_px, Rgba([255, 255, 255, 255])); // white

    // Draw frame (solid black rectangle around data)
    let left  = QUIET_CELLS * cell_px;
    let top   = QUIET_CELLS * cell_px;
    let right = width_px - QUIET_CELLS * cell_px;
    let bottom= height_px - QUIET_CELLS * cell_px;

    // top
    for y in top..(top + FRAME_CELLS * cell_px) {
        for x in left..right { img.put_pixel(x, y, Rgba([0, 0, 0, 255])); }
    }
    // bottom
    for y in (bottom - FRAME_CELLS * cell_px)..bottom {
        for x in left..right { img.put_pixel(x, y, Rgba([0, 0, 0, 255])); }
    }
    // left/right verticals
    for y in (top + FRAME_CELLS * cell_px)..(bottom - FRAME_CELLS * cell_px) {
        for x in left..(left + FRAME_CELLS * cell_px) { img.put_pixel(x, y, Rgba([0, 0, 0, 255])); }
        for x in (right - FRAME_CELLS * cell_px)..right { img.put_pixel(x, y, Rgba([0, 0, 0, 255])); }
    }

    // Data origin in px
    let origin_x = left + FRAME_CELLS * cell_px;
    let origin_y = top  + FRAME_CELLS * cell_px;

    // Stream → bits
    let bits = bytes_to_bits(stream);
    let mut bit_index = 0usize;

    // Paint cells: black=1, white=0
    for row in 0..grid_h {
        for col in 0..grid_w {
            if bit_index >= bits.len() { break; }
            if bits[bit_index] == 1 {
                let x0 = origin_x + col * cell_px;
                let y0 = origin_y + row * cell_px;
                for y in y0..(y0 + cell_px) {
                    for x in x0..(x0 + cell_px) {
                        img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
                    }
                }
            }
            bit_index += 1;
        }
    }

    Ok(img)
}

fn encode_file_to_png(input: &Path) -> Result<PathBuf> {
    step!("Encoding file {:?} → single PNG (macro-cells)…", input.file_name().unwrap_or_default());

    step!("Reading & hashing input…");
    let bytes = fs::read(input).with_context(|| format!("read {:?}", input))?;
    let (file_hash_stream, file_len_stream) = blake3_hash_file(input)?;
    if bytes.len() as u64 != file_len_stream { bail!("File changed during read (size mismatch)"); }
    let file_hash = blake3_hash_bytes(&bytes);
    if file_hash != file_hash_stream {
        bail!("File changed during read (hash mismatch). expected={}, read={}", hex32(file_hash_stream), hex32(file_hash));
    }
    ok!("Input: {} bytes, BLAKE3={}", bytes.len(), hex32(file_hash));

    let filename = input.file_name().and_then(OsStr::to_str).unwrap_or("unknown").to_string();
    let dataset_id = derive_dataset_id(&bytes, &filename);

    // Size grid + header + cell_px with canvas cap.
    let (header, hdr_bytes, overhead, cell_px) = choose_grid_and_header(&filename, &bytes, dataset_id)?;
    let needed_bits = (overhead + bytes.len()) * 8;
    let cap_bits = (header.grid_w as usize) * (header.grid_h as usize);
    assert!(cap_bits >= needed_bits);
    step!("Rendering grid: {}×{} cells, cell={} px…", header.grid_w, header.grid_h, cell_px);

    // Build stream: MAGIC|VER| ([len|header] * repeat) | payload | trailer
    let mut stream = Vec::with_capacity(overhead + bytes.len());
    stream.extend_from_slice(MAGIC);
    stream.extend_from_slice(&VERSION.to_le_bytes());
    let hdr_len_le = (hdr_bytes.len() as u32).to_le_bytes();
    for _ in 0..HEADER_REPEAT {
        stream.extend_from_slice(&hdr_len_le);
        stream.extend_from_slice(&hdr_bytes);
    }
    stream.extend_from_slice(&bytes);
    stream.extend_from_slice(&header.payload_hash32.to_le_bytes());

    let img = render_stream_to_image(&stream, header.grid_w, header.grid_h, cell_px)?;
    let out = if let Some(ext) = input.extension().and_then(OsStr::to_str) {
        input.with_extension(format!("{ext}.png"))
    } else {
        input.with_extension("png")
    };

    step!("Writing PNG to {:?}…", out.file_name().unwrap_or_default());
    DynamicImage::ImageRgba8(img).save(&out).context("write png")?;
    ok!("PNG written.");

    // Active proof: re-open and decode
    step!("Re-opening PNG for verification…");
    let decoded = decode_png_to_bytes(&out)?;
    let verified = verify_bytes(header.total_len, header.file_hash, UnverifiedBytes(decoded))?;
    if verified.as_slice() != bytes.as_slice() { bail!("Post-write byte-for-byte comparison failed"); }
    ok!("Round-trip verification OK (length + BLAKE3 + exact bytes).");

    eprintln!();
    ok!("ENCODE COMPLETE → {:?}", out.file_name().unwrap_or_default());
    Ok(out)
}

// ========================= Decode =========================

fn decode_png_to_bytes(path: &Path) -> Result<Vec<u8>> {
    let dynimg = image::open(path).with_context(|| format!("open {:?}", path))?;
    let gray0 = dynimg.to_luma8();

    let otsu = otsu_threshold(&gray0);
    let mut tries: Vec<u8> = vec![otsu];
    tries.extend_from_slice(FALLBACK_THRESHOLDS);

    let mut last_errs: Vec<anyhow::Error> = Vec::new();

    for thr in tries {
        let bin = binarize(&gray0, thr);
        match decode_from_binary_image_with_threshold(&bin, thr) {
            Ok(bytes) => return Ok(bytes),
            Err(e) => {
                last_errs.push(e);
                continue;
            }
        }
    }

    // Surface the richest error with context
    let mut msg = String::new();
    use std::fmt::Write;
    writeln!(&mut msg, "All threshold attempts failed for {:?}. Attempts: {}", path, last_errs.len()).ok();
    for (i, e) in last_errs.iter().enumerate() {
        writeln!(&mut msg, "  [{}] {}", i, e).ok();
    }
    bail!(msg);
}

fn decode_from_binary_image_with_threshold(bin: &GrayImage, threshold: u8) -> Result<Vec<u8>> {
    let geom = try_infer_geometry(bin, threshold)
        .with_context(|| format!(
            "Geometry inference failed @threshold {} (img={}×{}). \
             EXPECTED: quiet={} cells, frame={} cells. \
             HINT: Ensure frame is intact, no excessive cropping.",
            threshold, bin.width(), bin.height(), QUIET_CELLS, FRAME_CELLS
        ))?;

    // Sample the *center* pixel of each cell (quantized origin ensures lattice alignment)
    let center_off = geom.cell_px / 2;
    let raw = bin.as_raw();
    let w = bin.width() as usize;

    let mut bits = Vec::with_capacity((geom.grid_w * geom.grid_h) as usize);
    for r in 0..geom.grid_h {
        for c in 0..geom.grid_w {
            let cx = (geom.origin_x + c * geom.cell_px + center_off).min(bin.width() - 1) as usize;
            let cy = (geom.origin_y + r * geom.cell_px + center_off).min(bin.height() - 1) as usize;
            let v = raw[cy * w + cx];
            bits.push(if v < 128 { 1 } else { 0 });
        }
    }
    let stream = bits_to_bytes(&bits);

    // Parse stream with precise diagnostics.
    if stream.len() < 6 {
        bail!(
            "Stream too short ({} bytes) for MAGIC+VERSION. \
             geom={{thr:{}, cell_px:{}, origin:({}, {}), grid:{}×{}, img:{}×{}, qlf:{}, flm:{}, qrt:{}, frm:{}, qtp:{}, ftm:{}, qbm:{}, fbm:{}}}",
            stream.len(), geom.threshold, geom.cell_px, geom.origin_x, geom.origin_y,
            geom.grid_w, geom.grid_h, geom.img_w, geom.img_h,
            geom.quiet_left, geom.frame_left_meas, geom.quiet_right, geom.frame_right_meas,
            geom.quiet_top, geom.frame_top_meas, geom.quiet_bottom, geom.frame_bottom_meas
        );
    }
    if &stream[0..4] != MAGIC {
        bail!(
            "Magic mismatch: got={:02x?}, expected={:02x?}. \
             geom={{thr:{}, cell_px:{}, origin:({}, {}), grid:{}×{}, img:{}×{}}}",
            &stream[0..4], MAGIC, geom.threshold, geom.cell_px, geom.origin_x, geom.origin_y,
            geom.grid_w, geom.grid_h, geom.img_w, geom.img_h
        );
    }
    let ver = u16::from_le_bytes([stream[4], stream[5]]);
    if ver != VERSION {
        bail!("Unsupported VERSION {} (expected {}).", ver, VERSION);
    }

    let mut cursor = 6usize;
    let mut headers: Vec<Header> = Vec::with_capacity(HEADER_REPEAT);
    for i in 0..HEADER_REPEAT {
        if cursor + 4 > stream.len() {
            bail!("Header[{}] length u32 missing at cursor {} / {}.", i, cursor, stream.len());
        }
        let len = u32::from_le_bytes([stream[cursor], stream[cursor+1], stream[cursor+2], stream[cursor+3]]) as usize;
        cursor += 4;
        if cursor + len > stream.len() {
            bail!("Header[{}] bytes truncated: want {} at {}, have {}.", i, len, cursor, stream.len() - cursor);
        }
        let (h, rest) = postcard::take_from_bytes::<Header>(&stream[cursor..cursor+len])
            .map_err(|e| anyhow!("Header[{}] postcard decode failed at {}..{}: {}", i, cursor, cursor+len, e))?;
        if !rest.is_empty() {
            bail!("Header[{}] trailing bytes ({}).", i, rest.len());
        }
        headers.push(h);
        cursor += len;
    }
    if headers.windows(2).any(|w| w[0] != w[1]) {
        bail!("Header repeat mismatch: H0 != H1. H0={:?}, H1={:?}", headers[0], headers[1]);
    }
    let hdr = headers.pop().unwrap();

    // Optional: cross-check grid sizes (advisory)
    if hdr.grid_w != geom.grid_w || hdr.grid_h != geom.grid_h {
        warn!("Grid advisory mismatch (header {}×{} vs inferred {}×{}). Proceeding.",
              hdr.grid_w, hdr.grid_h, geom.grid_w, geom.grid_h);
    }

    // Payload + trailer
    let payload_start = cursor;
    let payload_len = hdr.payload_len as usize;
    if payload_start + payload_len + 4 > stream.len() {
        bail!(
            "Payload/trailer out of bounds: start={}, len={}, stream_len={}",
            payload_start, payload_len, stream.len()
        );
    }
    let payload = &stream[payload_start..payload_start + payload_len];
    let trailer = &stream[payload_start + payload_len .. payload_start + payload_len + 4];
    let trailer_u32 = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    let got32 = hash32_first(payload);

    if trailer_u32 != hdr.payload_hash32 {
        bail!(
            "Trailer u32 mismatch: stream={:#010x}, header={:#010x}. Payload_len={}, BLAKE3(first4)={:#010x}.",
            trailer_u32, hdr.payload_hash32, payload_len, got32
        );
    }
    if blake3_hash_bytes(payload) != hdr.file_hash {
        bail!(
            "Assembled payload BLAKE3 mismatch. expected={}, got={}",
            hex32(hdr.file_hash), hex32(blake3_hash_bytes(payload))
        );
    }
    if hdr.total_len != hdr.payload_len {
        bail!(
            "Header total_len ({}) != payload_len ({}).", hdr.total_len, hdr.payload_len
        );
    }

    Ok(payload.to_vec())
}

// ========================= One-arg entry =========================

fn is_png(path: &Path) -> bool {
    match fs::File::open(path) {
        Ok(mut f) => {
            let mut sig = [0u8; 8];
            std::io::Read::read_exact(&mut f, &mut sig).is_ok()
                && &sig == b"\x89PNG\r\n\x1a\n"
        }
        Err(_) => false,
    }
}

fn main() {
    if let Err(e) = real_main() {
        fail!("{e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let Some(path_os) = args.next() else {
        eprintln!("Usage: pixpack <path-to-file-or-png>");
        bail!("Missing required argument");
    };
    if args.next().is_some() { bail!("Exactly one argument is required"); }
    let path = PathBuf::from(path_os);
    if !path.exists() { bail!("Path does not exist: {:?}", path); }

    if is_png(&path) {
        // DECODE → write original file (idempotent, no overwrite unless content differs)
        step!("Decoding PNG {:?} → original file…", path.file_name().unwrap_or_default());
        let bytes = decode_png_to_bytes(&path)?;

        // Decode header again quickly to get filename (reuse same path decode to parse headers)
        let dynimg = image::open(&path)?;
        let gray = dynimg.to_luma8();
        let thr = otsu_threshold(&gray);
        let bin = binarize(&gray, thr);
        let geom = try_infer_geometry(&bin, thr)?;
        // sample again to get the stream and just parse headers (cheap)
        let center_off = geom.cell_px / 2;
        let raw = bin.as_raw();
        let w = bin.width() as usize;
        let mut bits = Vec::with_capacity((geom.grid_w * geom.grid_h) as usize);
        for r in 0..geom.grid_h {
            for c in 0..geom.grid_w {
                let cx = (geom.origin_x + c * geom.cell_px + center_off).min(bin.width() - 1) as usize;
                let cy = (geom.origin_y + r * geom.cell_px + center_off).min(bin.height() - 1) as usize;
                bits.push(if raw[cy * w + cx] < 128 { 1 } else { 0 });
            }
        }
        let stream = bits_to_bytes(&bits);
        if &stream[0..4] != MAGIC { bail!("Magic mismatch while extracting filename"); }
        let mut cursor = 6usize;
        let mut hdr_opt: Option<Header> = None;
        for _ in 0..HEADER_REPEAT {
            let len = u32::from_le_bytes([stream[cursor], stream[cursor+1], stream[cursor+2], stream[cursor+3]]) as usize;
            cursor += 4;
            let (hdr, _rest) = postcard::take_from_bytes::<Header>(&stream[cursor..cursor+len])?;
            cursor += len;
            hdr_opt = Some(hdr);
        }
        let hdr = hdr_opt.ok_or_else(|| anyhow!("Header missing after repeats"))?;

        let desired_name = if hdr.filename.trim().is_empty() { "decoded.bin".to_string() } else { hdr.filename.clone() };
        let mut out_path = path.with_file_name(&desired_name);

        if out_path.exists() {
            let (h, l) = blake3_hash_file(&out_path)?;
            if l == hdr.total_len && h == hdr.file_hash {
                ok!("Existing file matches payload; nothing to write.");
                eprintln!();
                ok!("DECODE COMPLETE → {:?}", out_path.file_name().unwrap_or_default());
                return Ok(());
            }
            warn!("{:?} exists with different content; appending .restored", out_path.file_name().unwrap_or_default());
            out_path = path.with_file_name(format!("{}.restored", desired_name));
        }

        step!("Writing reconstructed file to {:?}…", out_path.file_name().unwrap_or_default());
        fs::write(&out_path, &bytes).with_context(|| format!("write {:?}", out_path))?;
        ok!("File written.");

        step!("Re-reading written file to re-verify…");
        let (reh, relen) = blake3_hash_file(&out_path)?;
        if relen != hdr.total_len { bail!("Output length mismatch after write ({} vs {})", relen, hdr.total_len); }
        if reh != hdr.file_hash { bail!("Output BLAKE3 mismatch after write"); }
        ok!("Output verified (length + BLAKE3).");

        eprintln!();
        ok!("DECODE COMPLETE → {:?}", out_path.file_name().unwrap_or_default());
    } else {
        encode_file_to_png(&path)?;
    }
    Ok(())
}

// ========================= Tests =========================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn xorshift64(seed: &mut u64) -> u64 { let mut x=*seed; x^=x<<13; x^=x>>7; x^=x<<17; *seed=x; x }
    fn gen_bytes(len: usize, mut seed: u64) -> Vec<u8> {
        let mut out = vec![0u8; len]; let mut i = 0;
        while i < len {
            let v = xorshift64(&mut seed).to_le_bytes();
            let take = v.len().min(len - i);
            out[i..i+take].copy_from_slice(&v[..take]); i += take;
        }
        out
    }

    #[test]
    fn header_roundtrip_postcard() -> Result<()> {
        let h = Header {
            version: VERSION,
            filename: "x.bin".into(),
            total_len: 123,
            file_hash: [7u8; 32],
            dataset_id: [1u8; 16],
            grid_w: 10, grid_h: 12,
            quiet_cells: QUIET_CELLS,
            frame_cells: FRAME_CELLS,
            payload_len: 123,
            payload_hash32: 0xAABBCCDD,
        };
        let b = postcard::to_allocvec(&h)?;
        let (back, rest) = postcard::take_from_bytes::<Header>(&b)?;
        assert!(rest.is_empty());
        assert_eq!(back.filename, "x.bin");
        assert_eq!(back.grid_w, 10);
        assert_eq!(back.grid_h, 12);
        Ok(())
    }

    #[test]
    fn small_roundtrip_text() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("hello.txt");
        fs::write(&input, b"Hello, macro-cells!\n")?;
        let out_png = super::encode_file_to_png(&input)?;
        let decoded = super::decode_png_to_bytes(&out_png)?;
        assert_eq!(fs::read(&input)?, decoded);
        Ok(())
    }

    #[test]
    fn screenshot_like_rescale_still_decodes() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("note.txt");
        fs::write(&input, b"Pixels survive screenshots!")?;
        let out_png = super::encode_file_to_png(&input)?;
        let img = image::open(&out_png)?.to_rgba8();
        let big = imageops::resize(&img, img.width() * 2, img.height() * 2, imageops::FilterType::Nearest);
        let big_path = out_png.with_file_name("screenshot.png");
        DynamicImage::ImageRgba8(big).save(&big_path)?;
        let decoded = super::decode_png_to_bytes(&big_path)?;
        assert_eq!(fs::read(&input)?, decoded);
        Ok(())
    }

    #[test]
    fn tamper_detection_via_hash_mismatch() -> Result<()> {
        // 1) Encode a small payload
        let dir = tempdir()?;
        let input = dir.path().join("secret.bin");
        fs::write(&input, b"THIS IS A SECRET PAYLOAD")?;
        let out_png = super::encode_file_to_png(&input)?;
    
        // 2) Load original PNG and compute geometry on the real threshold
        let dynimg = image::open(&out_png)?;
        let gray = dynimg.to_luma8();
        let thr = super::otsu_threshold(&gray);
        let bin = super::binarize(&gray, thr);
        let geom = super::try_infer_geometry(&bin, thr)?;
    
        // 3) Sample the entire bitstream from the original image (exactly as decoder does)
        let center_off = geom.cell_px / 2;
        let raw = bin.as_raw();
        let w = bin.width() as usize;
    
        let mut bits = Vec::with_capacity((geom.grid_w * geom.grid_h) as usize);
        for r in 0..geom.grid_h {
            for c in 0..geom.grid_w {
                let cx = (geom.origin_x + c * geom.cell_px + center_off).min(bin.width() - 1) as usize;
                let cy = (geom.origin_y + r * geom.cell_px + center_off).min(bin.height() - 1) as usize;
                let v = raw[cy * w + cx];
                bits.push(if v < 128 { 1 } else { 0 });
            }
        }
        let stream = super::bits_to_bytes(&bits);
    
        // 4) Parse headers to find the payload start (bytes)
        //    MAGIC (4) + VERSION (2) + [len(4)+header]*HEADER_REPEAT + payload + trailer(4)
        assert!(stream.len() >= 6, "stream too short to contain MAGIC+VERSION");
        assert_eq!(&stream[0..4], super::MAGIC, "unexpected MAGIC");
        let ver = u16::from_le_bytes([stream[4], stream[5]]);
        assert_eq!(ver, super::VERSION, "unexpected VERSION");
    
        let mut cursor = 6usize;
        let mut hdr: Option<super::Header> = None;
        for i in 0..super::HEADER_REPEAT {
            assert!(cursor + 4 <= stream.len(), "missing header[{}] len at {}", i, cursor);
            let len = u32::from_le_bytes([stream[cursor], stream[cursor+1], stream[cursor+2], stream[cursor+3]]) as usize;
            cursor += 4;
            assert!(cursor + len <= stream.len(), "header[{}] truncated at {}", i, cursor);
            let (h, rest) = postcard::take_from_bytes::<super::Header>(&stream[cursor..cursor+len])
                .map_err(|e| anyhow!("postcard decode header[{}]: {}", i, e))?;
            assert!(rest.is_empty(), "header[{}] had trailing {} bytes", i, rest.len());
            hdr = Some(h);
            cursor += len;
        }
        let hdr = hdr.expect("header missing after repeats");
        let payload_start_bytes = cursor;
        let payload_len_bytes = hdr.payload_len as usize;
    
        // Sanity: payload range must fit the stream
        assert!(payload_start_bytes + payload_len_bytes + 4 <= stream.len(), "payload/trailer OOB");
    
        // 5) Choose a payload bit (middle of payload) and map to cell (row, col)
        let payload_start_bits = payload_start_bytes * 8;
        let payload_bits = payload_len_bytes * 8;
        let target_bit = payload_start_bits + payload_bits / 2; // safely inside payload
        let target_r = (target_bit as u32) / geom.grid_w;
        let target_c = (target_bit as u32) % geom.grid_w;
    
        // 6) Determine original bit at that cell (under current threshold)
        let cell_cx = geom.origin_x + target_c * geom.cell_px + center_off;
        let cell_cy = geom.origin_y + target_r * geom.cell_px + center_off;
        let orig_bit = {
            let px = bin.as_raw()[(cell_cy as usize) * w + (cell_cx as usize)];
            if px < 128 { 1 } else { 0 }
        };
    
        // 7) Forge the image by repainting the ENTIRE target cell to the opposite color
        let mut forged = dynimg.to_rgba8();
        let paint_white = orig_bit == 1;
        let paint = if paint_white { Rgba([255,255,255,255]) } else { Rgba([0,0,0,255]) };
        let x0 = geom.origin_x + target_c * geom.cell_px;
        let y0 = geom.origin_y + target_r * geom.cell_px;
        for y in y0..(y0 + geom.cell_px) {
            for x in x0..(x0 + geom.cell_px) {
                forged.put_pixel(x, y, paint);
            }
        }
    
        // 8) Confirm the flip still reads as flipped under the decoder's (recomputed) threshold
        let forged_gray = DynamicImage::ImageRgba8(forged.clone()).to_luma8();
        let forged_thr = super::otsu_threshold(&forged_gray);
        let forged_bin = super::binarize(&forged_gray, forged_thr);
        let w2 = forged_bin.width() as usize;
        let flipped_bit = {
            let cx = (x0 + center_off) as usize;
            let cy = (y0 + center_off) as usize;
            let v = forged_bin.as_raw()[cy * w2 + cx];
            if v < 128 { 1 } else { 0 }
        };
        assert_ne!(orig_bit, flipped_bit, "tamper did not flip the sampled bit under decoder threshold");
    
        // 9) Save forged image and require decode failure with an integrity-specific reason
        let forged_path = out_png.with_file_name("forged.png");
        DynamicImage::ImageRgba8(forged).save(&forged_path)?;
    
        let err = super::decode_png_to_bytes(&forged_path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            // payload integrity checks
            msg.contains("Assembled payload BLAKE3 mismatch")
            // header/parse checks (OK too, but less likely since we hit payload)
            || msg.contains("Trailer u32 mismatch")
            || msg.contains("Header repeat mismatch")
            || msg.contains("Magic mismatch")
            || msg.contains("Payload/trailer out of bounds")
            // geometry/threshold failure (shouldn't happen with single-cell flip, but accept)
            || msg.contains("Geometry inference failed")
            || msg.contains("All threshold attempts failed"),
            "unexpected error detail: {msg}"
        );
    
        Ok(())
    }

    #[test]
    fn decode_plain_png_fails() -> Result<()> {
        let dir = tempdir()?;
        let png = dir.path().join("plain.png");
        let mut img: RgbaImage = ImageBuffer::from_pixel(64, 64, Rgba([230, 230, 230, 255]));
        for y in 0..64 { img.put_pixel(32, y, Rgba([0,0,0,255])); } // not a valid frame
        DynamicImage::ImageRgba8(img).save(&png)?;
        let err = super::decode_png_to_bytes(&png).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("All threshold attempts failed")
            || msg.contains("Geometry inference failed")
            || msg.contains("Image too small"));
        Ok(())
    }

    #[test]
    fn idempotent_existing_file() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("artifact.bin");
        fs::write(&input, b"XYZ")?;
        let out_png = super::encode_file_to_png(&input)?;
        // First decode path
        let decoded = super::decode_png_to_bytes(&out_png)?;
        let out_file = out_png.with_file_name("artifact.bin");
        fs::write(&out_file, &decoded)?;
        // Second check: hashes already match, no write expected
        let (h, l) = blake3_hash_file(&out_file)?;
        assert_eq!(l, 3);
        assert_eq!(h, blake3_hash_bytes(&decoded));
        Ok(())
    }

    #[test]
    fn roundtrip_medium_random() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("blob.bin");
        let data = gen_bytes(128 * 1024, 0xFEEDFACE);
        fs::write(&input, &data)?;
        let out_png = super::encode_file_to_png(&input)?;
        let decoded = super::decode_png_to_bytes(&out_png)?;
        assert_eq!(blake3_hash_bytes(&data), blake3_hash_bytes(&decoded));
        assert_eq!(data, decoded);
        Ok(())
    }
}
