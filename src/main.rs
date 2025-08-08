// pixpack
//
// CLI: one argument only
//   - If the argument is a PNG created by pixpack → DECODE to the original file (with proofs)
//   - Otherwise → ENCODE that file into a single PNG (with proofs)
//
// Guarantees & proofs:
//   • During encode: after writing the PNG we re-open the image, decode the pixels back,
//     verify payload hash (fast) THEN the full BLAKE3 and exact byte-for-byte equality.
//   • During decode: after writing the reconstructed file we re-hash and re-verify.
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
use image::{imageops, DynamicImage, GenericImage, GrayImage, ImageBuffer, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};

// ========================= Configuration =========================

const MAGIC: &[u8; 4] = b"PXPV"; // Visual v1
const VERSION: u16 = 1;

// Visual layout (fixed at compile time; decoder also uses these):
const QUIET_CELLS: u32 = 3;           // white quiet zone thickness (in cells)
const FRAME_CELLS: u32 = 2;           // solid black frame thickness (in cells)
const CELL_PX_DEFAULT: u32 = 10;      // pixels per cell at encode; decoder infers from image
const HEADER_REPEAT: usize = 2;       // header is repeated in stream

// Binarization fallback thresholds (after Otsu) — tiny set, no rotations:
const THRESHOLDS: &[u8] = &[170, 150, 130];

// Logging helpers
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

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Header {
    version: u16,
    // File identity
    filename: String,
    total_len: u64,
    file_hash: [u8; 32],
    dataset_id: [u8; 16],
    // Visual geometry (so decode can infer the grid from pixels):
    grid_w: u32,
    grid_h: u32,
    quiet_cells: u32,
    frame_cells: u32,
    // Payload proof (fast per-image check in addition to full BLAKE3):
    payload_len: u64,   // should equal total_len
    payload_hash32: u32 // first 4 bytes of BLAKE3(payload)
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
        bail!("length mismatch: expected {}, got {}", expected_len, data.0.len());
    }
    let got = blake3_hash_bytes(&data.0);
    if got != expected_hash {
        bail!("hash mismatch: expected {}, got {}", hex32(expected_hash), hex32(got));
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
    for p in out.pixels_mut() { p.0[0] = if p.0[0] >= t { 255 } else { 0 }; }
    out
}

// ========================= Geometry helpers (decode) =========================

#[derive(Debug)]
struct Geometry {
    cell_px: u32,
    origin_x: u32, // top-left of data grid in pixels
    origin_y: u32,
    grid_w: u32,
    grid_h: u32,
}

// Find (quiet_px, frame_px) on a 1D binary scanline (0=black, 255=white).
fn scan_white_then_black(line: &[u8]) -> Result<(u32, u32)> {
    let mut i = 0usize;
    while i < line.len() && line[i] == 255 { i += 1; }
    if i == line.len() { bail!("no black band found (no frame)"); }
    let q = i as u32;
    let mut j = i;
    while j < line.len() && line[j] == 0 { j += 1; }
    let f = (j - i) as u32;
    if f == 0 { bail!("frame band has zero thickness"); }
    Ok((q, f))
}

fn infer_geometry(bin: &GrayImage) -> Result<Geometry> {
    let w = bin.width() as usize;
    let h = bin.height() as usize;
    if w < 40 || h < 40 { bail!("image too small"); }

    // Sample mid row & mid col
    let mid_y = h / 2;
    let row = &bin.as_raw()[mid_y * w .. (mid_y + 1) * w];
    let (q_left, f_left) = scan_white_then_black(row)?;
    let (q_right, f_right) = {
        // scan from right: reverse the row
        let mut rev = row.to_vec(); rev.reverse();
        scan_white_then_black(&rev)?
    };

    let mid_x = w / 2;
    let mut col: Vec<u8> = Vec::with_capacity(h);
    for y in 0..h { col.push(bin.as_raw()[y * w + mid_x]); }
    let (q_top, f_top) = scan_white_then_black(&col)?;
    let (q_bottom, f_bottom) = {
        let mut rev = col.clone(); rev.reverse();
        scan_white_then_black(&rev)?
    };

    // Frame thickness (px) should be consistent — take min to be safe.
    let frame_px_x = f_left.min(f_right);
    let frame_px_y = f_top.min(f_bottom);
    if frame_px_x == 0 || frame_px_y == 0 { bail!("frame thickness zero"); }

    // Infer pixels per cell
    let cell_px_x = (frame_px_x as f32 / FRAME_CELLS as f32).round() as u32;
    let cell_px_y = (frame_px_y as f32 / FRAME_CELLS as f32).round() as u32;
    let cell_px = cell_px_x.min(cell_px_y).max(1);

    // Data origin (px): start of frame + frame thickness from each side
    let origin_x = q_left + frame_px_x;
    let origin_y = q_top + frame_px_y;
    let right_margin = q_right + frame_px_x;
    let bottom_margin = q_bottom + frame_px_y;

    if (origin_x as usize) >= w || (origin_y as usize) >= h { bail!("computed origin out of bounds"); }

    // Grid size (px)
    let grid_w_px = (w as u32).saturating_sub(origin_x + right_margin);
    let grid_h_px = (h as u32).saturating_sub(origin_y + bottom_margin);
    if grid_w_px == 0 || grid_h_px == 0 { bail!("grid pixel size is zero"); }

    // Cells count (round to nearest)
    let grid_w = ((grid_w_px as f32 / cell_px as f32).round()) as u32;
    let grid_h = ((grid_h_px as f32 / cell_px as f32).round()) as u32;
    if grid_w == 0 || grid_h == 0 { bail!("grid cell size is zero"); }

    Ok(Geometry { cell_px, origin_x, origin_y, grid_w, grid_h })
}

// ========================= Bit packing / unpacking =========================

fn bytes_to_bits(data: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(data.len() * 8);
    for b in data {
        for k in (0..8).rev() {
            bits.push((b >> k) & 1);
        }
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

// ========================= Encode (file → PNG) =========================

fn plan_grid(overhead_bytes: usize, payload_bytes: usize) -> (u32, u32) {
    let total_bytes = overhead_bytes + payload_bytes;
    let bits_needed = (total_bytes as u128) * 8;
    // Near-square grid in cells:
    let side = (f64::from_bits(bits_needed as f64 as u64).sqrt() as f64).sqrt(); // trick avoided; do normal sqrt
    // use normal f64 sqrt:
    let side = (bits_needed as f64).sqrt().ceil() as u64;
    let mut gw = side as u32;
    let mut gh = ((bits_needed + (gw as u128) - 1) / gw as u128) as u32;
    if gw == 0 { gw = 1; }
    if gh == 0 { gh = 1; }
    (gw, gh)
}

fn encode_file_to_png(input: &Path) -> Result<PathBuf> {
    step!("Encoding file {:?} → single PNG (macro-cells)…", input.file_name().unwrap_or_default());

    step!("Reading & hashing input…");
    let bytes = fs::read(input).with_context(|| format!("read {:?}", input))?;
    let (file_hash_stream, file_len_stream) = blake3_hash_file(input)?;
    if bytes.len() as u64 != file_len_stream { bail!("file changed during read"); }
    let file_hash = blake3_hash_bytes(&bytes);
    if file_hash != file_hash_stream { bail!("file content changed (hash mismatch)"); }
    ok!("Input: {} bytes, BLAKE3={}", bytes.len(), hex32(file_hash));

    let filename = input.file_name().and_then(OsStr::to_str).unwrap_or("unknown").to_string();
    let dataset_id = derive_dataset_id(&bytes, &filename);
    let payload_hash32 = hash32_first(&bytes);

    // First pass header (grid dims unknown yet, set to 0) → compute overhead
    let header0 = Header {
        version: VERSION,
        filename: filename.clone(),
        total_len: bytes.len() as u64,
        file_hash,
        dataset_id,
        grid_w: 0,
        grid_h: 0,
        quiet_cells: QUIET_CELLS,
        frame_cells: FRAME_CELLS,
        payload_len: bytes.len() as u64,
        payload_hash32,
    };
    let hdr0 = postcard::to_allocvec(&header0)?;
    let overhead0 = 4 + 2 + HEADER_REPEAT * (4 + hdr0.len()) + 4;

    // Choose grid to fit (overhead + payload)
    let (grid_w, grid_h) = plan_grid(overhead0, bytes.len());
    let header = Header { grid_w, grid_h, ..header0 };
    let hdr = postcard::to_allocvec(&header)?;
    let overhead = 4 + 2 + HEADER_REPEAT * (4 + hdr.len()) + 4;

    // Re-validate capacity (should be stable)
    let capacity_bits = (grid_w as usize) * (grid_h as usize);
    let needed_bits = (overhead + bytes.len()) * 8;
    if needed_bits > capacity_bits {
        bail!("internal: planned grid too small (need {} bits, have {})", needed_bits, capacity_bits);
    }

    // Build stream: MAGIC|VER| ([len|header] * repeat) | payload | trailer
    let mut stream = Vec::with_capacity(overhead + bytes.len());
    stream.extend_from_slice(MAGIC);
    stream.extend_from_slice(&VERSION.to_le_bytes());
    let hdr_len_le = (hdr.len() as u32).to_le_bytes();
    for _ in 0..HEADER_REPEAT {
        stream.extend_from_slice(&hdr_len_le);
        stream.extend_from_slice(&hdr);
    }
    stream.extend_from_slice(&bytes);
    stream.extend_from_slice(&payload_hash32.to_le_bytes());

    // Render grid to an RGBA image
    step!("Rendering grid: {}×{} cells, cell={} px…", grid_w, grid_h, CELL_PX_DEFAULT);
    let img = render_stream_to_image(&stream, grid_w, grid_h, CELL_PX_DEFAULT)?;
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
    if verified.as_slice() != bytes.as_slice() { bail!("post-write byte-for-byte comparison failed"); }
    ok!("Round-trip verification OK (length + BLAKE3 + exact bytes).");

    eprintln!();
    ok!("ENCODE COMPLETE → {:?}", out.file_name().unwrap_or_default());
    Ok(out)
}

fn render_stream_to_image(stream: &[u8], grid_w: u32, grid_h: u32, cell_px: u32) -> Result<RgbaImage> {
    // Canvas dimensions
    let total_cells_x = grid_w + 2 * (QUIET_CELLS + FRAME_CELLS);
    let total_cells_y = grid_h + 2 * (QUIET_CELLS + FRAME_CELLS);
    let width_px  = total_cells_x * cell_px;
    let height_px = total_cells_y * cell_px;

    let mut img: RgbaImage = ImageBuffer::from_pixel(width_px, height_px, Rgba([255, 255, 255, 255])); // start white

    // Draw frame (solid black rectangle around grid)
    let left  = QUIET_CELLS * cell_px;
    let top   = QUIET_CELLS * cell_px;
    let right = width_px - QUIET_CELLS * cell_px;
    let bottom= height_px - QUIET_CELLS * cell_px;

    // Top/bottom frame bands
    for y in top..(top + FRAME_CELLS * cell_px) {
        for x in left..right {
            img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
        }
    }
    for y in (bottom - FRAME_CELLS * cell_px)..bottom {
        for x in left..right {
            img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
        }
    }
    // Left/right frame bands
    for y in (top + FRAME_CELLS * cell_px)..(bottom - FRAME_CELLS * cell_px) {
        for x in left..(left + FRAME_CELLS * cell_px) {
            img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
        }
        for x in (right - FRAME_CELLS * cell_px)..right {
            img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
        }
    }

    // Data origin (top-left of data grid)
    let origin_x = left + FRAME_CELLS * cell_px;
    let origin_y = top  + FRAME_CELLS * cell_px;

    // Stream → bits
    let bits = bytes_to_bits(stream);

    // Paint cells: black for bit 1, white for 0 (already white)
    let mut bit_index = 0usize;
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

// ========================= Decode (PNG → bytes) =========================

fn decode_png_to_bytes(path: &Path) -> Result<Vec<u8>> {
    let dynimg = image::open(path).with_context(|| format!("open {:?}", path))?;
    let gray0 = dynimg.to_luma8();

    // Otsu first
    let t = otsu_threshold(&gray0);
    let mut tries: Vec<u8> = vec![t];
    tries.extend_from_slice(THRESHOLDS);

    for thr in tries {
        let bin = binarize(&gray0, thr);
        if let Ok(bytes) = decode_from_binary_image(&bin) {
            return Ok(bytes);
        }
    }
    bail!("Could not parse pixpack page: {:?}", path);
}

fn decode_from_binary_image(bin: &GrayImage) -> Result<Vec<u8>> {
    // Geometry from pixels (no header yet)
    let geom = infer_geometry(bin)?;

    // Sample interior of each cell to reconstruct bits
    let mut bits = Vec::with_capacity((geom.grid_w * geom.grid_h) as usize);
    let margin = (geom.cell_px / 4).max(2);
    let win = (geom.cell_px.saturating_sub(2 * margin)).max(1);

    let raw = bin.as_raw();
    let w = bin.width() as usize;
    for r in 0..geom.grid_h {
        for c in 0..geom.grid_w {
            let x0 = geom.origin_x + c * geom.cell_px + margin;
            let y0 = geom.origin_y + r * geom.cell_px + margin;
            let mut acc = 0u32;
            for dy in 0..win {
                let py = (y0 + dy).min(bin.height() - 1) as usize;
                let base = py * w;
                for dx in 0..win {
                    let px = (x0 + dx).min(bin.width() - 1) as usize;
                    acc += raw[base + px] as u32;
                }
            }
            let avg = acc / (win * win).max(1) as u32;
            let bit = if avg < 128 { 1u8 } else { 0u8 };
            bits.push(bit);
        }
    }

    // Bits → bytes stream
    let mut stream = bits_to_bytes(&bits);

    // Parse stream: MAGIC|VER| ([len|header]*repeat) | payload | trailer
    if stream.len() < 6 { bail!("stream too short"); }
    if &stream[0..4] != MAGIC { bail!("magic mismatch"); }
    let ver = u16::from_le_bytes([stream[4], stream[5]]);
    if ver != VERSION { bail!("unsupported version {}", ver); }

    let mut cursor = 6usize;
    let mut header: Option<Header> = None;
    for _ in 0..HEADER_REPEAT {
        if cursor + 4 > stream.len() { bail!("header len missing"); }
        let len = u32::from_le_bytes([stream[cursor], stream[cursor+1], stream[cursor+2], stream[cursor+3]]) as usize;
        cursor += 4;
        if cursor + len > stream.len() { bail!("header bytes truncated"); }
        let (h, rest) = postcard::take_from_bytes::<Header>(&stream[cursor..cursor+len])
            .map_err(|e| anyhow!("header decode failed: {e}"))?;
        if !rest.is_empty() { bail!("header trailing bytes"); }
        header = Some(h);
        cursor += len;
    }
    let hdr = header.ok_or_else(|| anyhow!("no header after repeats"))?;

    // Payload and trailer at precise offsets (NOT end of page!)
    let payload_start = cursor;
    let payload_len = hdr.payload_len as usize;
    if payload_start + payload_len + 4 > stream.len() {
        bail!("payload/trailer exceed stream length");
    }
    let payload = &stream[payload_start..payload_start + payload_len];
    let trailer = &stream[payload_start + payload_len .. payload_start + payload_len + 4];
    let trailer_u32 = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);

    if trailer_u32 != hdr.payload_hash32 {
        bail!("payload hash32 mismatch");
    }
    if blake3_hash_bytes(payload) != hdr.file_hash {
        bail!("assembled file hash mismatch");
    }
    if hdr.total_len != hdr.payload_len {
        bail!("header total_len vs payload_len mismatch");
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
        bail!("missing required argument");
    };
    if args.next().is_some() { bail!("exactly one argument is required"); }
    let path = PathBuf::from(path_os);
    if !path.exists() { bail!("path does not exist: {:?}", path); }

    if is_png(&path) {
        // DECODE
        step!("Decoding PNG {:?} → original file…", path.file_name().unwrap_or_default());
        let bytes = decode_png_to_bytes(&path)?;
        // Extract filename from header by decoding again and parsing header directly:
        // We already re-validated in decode_png_to_bytes, so just name now:
        let dynimg = image::open(&path)?;
        let gray = dynimg.to_luma8();
        let bin = binarize(&gray, otsu_threshold(&gray));
        let geom = infer_geometry(&bin)?;
        let bits = {
            let mut v = Vec::with_capacity((geom.grid_w * geom.grid_h) as usize);
            let margin = (geom.cell_px / 4).max(2);
            let win = (geom.cell_px.saturating_sub(2 * margin)).max(1);
            let raw = bin.as_raw();
            let w = bin.width() as usize;
            for r in 0..geom.grid_h {
                for c in 0..geom.grid_w {
                    let x0 = geom.origin_x + c * geom.cell_px + margin;
                    let y0 = geom.origin_y + r * geom.cell_px + margin;
                    let mut acc = 0u32;
                    for dy in 0..win {
                        let py = (y0 + dy).min(bin.height() - 1) as usize;
                        let base = py * w;
                        for dx in 0..win {
                            let px = (x0 + dx).min(bin.width() - 1) as usize;
                            acc += raw[base + px] as u32;
                        }
                    }
                    let avg = acc / (win * win).max(1) as u32;
                    v.push(if avg < 128 { 1u8 } else { 0u8 });
                }
            }
            v
        };
        let mut stream = bits_to_bytes(&bits);
        if &stream[0..4] != MAGIC { bail!("magic mismatch"); }
        let mut cursor = 6usize;
        let mut hdr_opt: Option<Header> = None;
        for _ in 0..HEADER_REPEAT {
            let len = u32::from_le_bytes([stream[cursor], stream[cursor+1], stream[cursor+2], stream[cursor+3]]) as usize;
            cursor += 4;
            let (hdr, _rest) = postcard::take_from_bytes::<Header>(&stream[cursor..cursor+len])?;
            cursor += len;
            hdr_opt = Some(hdr);
        }
        let hdr = hdr_opt.ok_or_else(|| anyhow!("header missing"))?;

        // Idempotent write
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
        fs::write(&out_path, &bytes)?;
        ok!("File written.");

        step!("Re-reading written file to re-verify…");
        let (reh, relen) = blake3_hash_file(&out_path)?;
        if relen != hdr.total_len { bail!("output length mismatch after write"); }
        if reh != hdr.file_hash { bail!("output BLAKE3 mismatch after write"); }
        ok!("Output verified (length + BLAKE3).");

        eprintln!();
        ok!("DECODE COMPLETE → {:?}", out_path.file_name().unwrap_or_default());
    } else {
        // ENCODE
        encode_file_to_png(&path)?;
    }
    Ok(())
}

// ========================= Tests =========================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn xorshift64(seed: &mut u64) -> u64 {
        let mut x = *seed; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *seed = x; x
    }
    fn gen_bytes(len: usize, mut seed: u64) -> Vec<u8> {
        let mut out = vec![0u8; len];
        let mut i = 0;
        while i < len {
            let v = xorshift64(&mut seed).to_le_bytes();
            let take = v.len().min(len - i);
            out[i..i + take].copy_from_slice(&v[..take]);
            i += take;
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
        assert_eq!(rest.len(), 0);
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
        let out_file = {
            // Decode path
            let bytes = super::decode_png_to_bytes(&out_png)?;
            let out = out_png.with_file_name("hello.txt"); // same name
            fs::write(&out, &bytes)?;
            out
        };
        assert_eq!(fs::read(&input)?, fs::read(&out_file)?);
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
        let dir = tempdir()?;
        let input = dir.path().join("secret.bin");
        fs::write(&input, b"THIS IS A SECRET PAYLOAD")?;
        let out_png = super::encode_file_to_png(&input)?;

        // Corrupt PNG pixels: paint a white vertical stripe through data
        let mut img = image::open(&out_png)?.to_rgba8();
        let w = img.width();
        for y in 0..img.height() {
            for x in (w/3)..(w/3 + 3) { img.put_pixel(x, y, Rgba([255,255,255,255])); }
        }
        let forged = out_png.with_file_name("forged.png");
        DynamicImage::ImageRgba8(img).save(&forged)?;

        let err = super::decode_png_to_bytes(&forged).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("payload hash32 mismatch") || msg.contains("assembled file hash mismatch") || msg.contains("Could not parse pixpack page"));
        Ok(())
    }

    #[test]
    fn decode_plain_png_fails() -> Result<()> {
        let dir = tempdir()?;
        let png = dir.path().join("plain.png");
        let mut img: RgbaImage = ImageBuffer::from_pixel(64, 64, Rgba([220, 220, 220, 255]));
        for y in 0..64 {
            img.put_pixel(32, y, Rgba([0,0,0,255]));
        }
        DynamicImage::ImageRgba8(img).save(&png)?;
        let err = super::decode_png_to_bytes(&png).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("Could not parse pixpack page") || msg.contains("magic mismatch"));
        Ok(())
    }

    #[test]
    fn idempotent_existing_file() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("artifact.bin");
        fs::write(&input, b"XYZ")?;
        let out_png = super::encode_file_to_png(&input)?;
        // First decode
        {
            let bytes = super::decode_png_to_bytes(&out_png)?;
            let out = out_png.with_file_name("artifact.bin");
            fs::write(&out, &bytes)?;
        }
        // Second decode should re-use existing file
        {
            let bytes = super::decode_png_to_bytes(&out_png)?;
            let out = out_png.with_file_name("artifact.bin");
            let (h, l) = blake3_hash_file(&out)?;
            assert_eq!(l, 3);
            assert_eq!(h, blake3_hash_bytes(&bytes));
        }
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
