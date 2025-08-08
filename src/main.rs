// main.rs
// pixpack — visual encoding using tiled QR codes.
//
// Single-arg CLI:
//   - If the given path decodes as a pixpack image (any common format): DECODE to the original file.
//   - Otherwise: ENCODE the file into a *viewable* PNG made of QR codes.
// Every step includes *active proofs* (BLAKE3, length, and a post-write re-decode).
//
// Integrity framing per-QR:
//   magic:   b"PXP1"
//   kind:    0x01=Header | 0x02=Chunk
//   body:    postcard-serialized struct (VisualHeader/VisualChunk)
//
// VisualHeader { version, filename, total_len, file_hash, chunk_size, num_chunks }
// VisualChunk  { version, index, chunk_hash, data }
//
// Build: cargo build --release
// Run:   cargo run --release -- <path>

use std::{
    env,
    ffi::OsStr,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Context, Result};
use blake3::Hasher;
use image::{imageops, DynamicImage, GrayImage, ImageBuffer, Rgba, RgbaImage};
use qrcode::{types::Color, EcLevel, QrCode};
use quircs::Quirc;
use serde::{Deserialize, Serialize};

// -------------------- Constants & types --------------------

const MAGIC: &[u8; 4] = b"PXP1";
const VERSION: u16 = 1;

// Visual/layout parameters
const DEFAULT_CHUNK_SIZE: usize = 800;   // baseline; auto-tuned down if QR size too big
const MAX_MODULES_PER_QR: usize = 85;    // cap modules per side to stay readable/robust
const DEFAULT_MODULE_PX: u32 = 10;       // initial pixels per module
const MIN_MODULE_PX: u32 = 2;            // minimum pixels per module (avoid unreadable)
const TILE_MARGIN_PX: u32 = 20;          // white padding around each QR tile
const GRID_GAP_PX: u32 = 20;             // spacing between tiles
const HEADER_REPLICATION: usize = 3;     // repeat header QR for robustness
const QUIET_ZONE_MODULES: u32 = 4;       // additional quiet zone around QR content
const MAX_CANVAS_DIM: u32 = 8000;        // keep final PNG within this bound (both width & height)

// Decode fallbacks
const THRESHOLDS: &[u8] = &[180, 160, 140, 128, 110, 96];

#[derive(Debug, Clone)]
struct UnverifiedBytes(Vec<u8>);
#[derive(Debug, Clone)]
struct VerifiedBytes(Vec<u8>);
impl VerifiedBytes {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

// Frames carried in QR codes
#[derive(Debug, Serialize, Deserialize, Clone)]
struct VisualHeader {
    version: u16,
    filename: String,
    total_len: u64,
    file_hash: [u8; 32],
    chunk_size: u32,
    num_chunks: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct VisualChunk {
    version: u16,
    index: u32,
    chunk_hash: [u8; 32],
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
enum Frame {
    Header(VisualHeader),
    Chunk(VisualChunk),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameKind {
    Header = 0x01,
    Chunk = 0x02,
}

// pretty logging
macro_rules! step {
    ($($arg:tt)*) => { eprintln!("▶ {}", format!($($arg)*)); };
}
macro_rules! ok {
    ($($arg:tt)*) => { eprintln!("✔ {}", format!($($arg)*)); };
}
macro_rules! warn {
    ($($arg:tt)*) => { eprintln!("⚠ {}", format!($($arg)*)); };
}
macro_rules! fail {
    ($($arg:tt)*) => { eprintln!("✘ {}", format!($($arg)*)); };
}

// -------------------- Hashing & verification --------------------

fn blake3_hash_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(bytes);
    *h.finalize().as_bytes()
}

fn blake3_hash_file(path: &Path) -> Result<([u8; 32], u64)> {
    let mut f = fs::File::open(path).with_context(|| format!("open {:?}", path))?;
    let mut h = Hasher::new();
    let mut buf = vec![0u8; 1 << 20]; // 1MiB
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
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
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

// -------------------- Frame pack/unpack --------------------

fn pack_frame(frame: &Frame) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(1024);
    out.extend_from_slice(MAGIC);
    match frame {
        Frame::Header(h) => {
            out.push(FrameKind::Header as u8);
            let body = postcard::to_allocvec(h)?;
            out.extend_from_slice(&body);
        }
        Frame::Chunk(c) => {
            out.push(FrameKind::Chunk as u8);
            let body = postcard::to_allocvec(c)?;
            out.extend_from_slice(&body);
        }
    }
    Ok(out)
}

fn unpack_frame(payload: &[u8]) -> Result<Frame> {
    if payload.len() < MAGIC.len() + 1 {
        bail!("QR payload too short");
    }
    if &payload[..4] != MAGIC {
        bail!("QR magic mismatch");
    }
    let kind = payload[4];
    let body = &payload[5..];
    match kind {
        x if x == FrameKind::Header as u8 => {
            let (h, rest) = postcard::take_from_bytes::<VisualHeader>(body)
                .map_err(|e| anyhow!("header decode failed: {e}"))?;
            if !rest.is_empty() { bail!("trailing bytes after header"); }
            if h.version != VERSION { bail!("unsupported header version {}", h.version); }
            Ok(Frame::Header(h))
        }
        x if x == FrameKind::Chunk as u8 => {
            let (c, rest) = postcard::take_from_bytes::<VisualChunk>(body)
                .map_err(|e| anyhow!("chunk decode failed: {e}"))?;
            if !rest.is_empty() { bail!("trailing bytes after chunk"); }
            if c.version != VERSION { bail!("unsupported chunk version {}", c.version); }
            Ok(Frame::Chunk(c))
        }
        _ => bail!("unknown frame kind"),
    }
}

// -------------------- QR helpers --------------------

fn qr_for_bytes(data: &[u8]) -> Result<QrCode> {
    // Highest ECC for robustness; crate chooses smallest version that fits.
    QrCode::with_error_correction_level(data, EcLevel::H).map_err(|e| anyhow!("QR encode failed: {e}"))
}

fn render_qr_to_rgba(qr: &QrCode, module_px: u32, extra_quiet_zone_modules: u32) -> RgbaImage {
    let w = qr.width() as u32;
    let total_modules = w + extra_quiet_zone_modules * 2;
    let size_px = total_modules * module_px;

    let mut img: RgbaImage = ImageBuffer::from_pixel(size_px, size_px, Rgba([255, 255, 255, 255]));

    for y in 0..w {
        for x in 0..w {
            if qr[(x as usize, y as usize)] == Color::Dark {
                let x0 = (x + extra_quiet_zone_modules) * module_px;
                let y0 = (y + extra_quiet_zone_modules) * module_px;
                for yy in 0..module_px {
                    for xx in 0..module_px {
                        img.put_pixel(x0 + xx, y0 + yy, Rgba([0, 0, 0, 255]));
                    }
                }
            }
        }
    }
    img
}

fn choose_chunk_size(bytes: usize) -> usize {
    // Start at DEFAULT_CHUNK_SIZE and shrink until each chunk QR width <= MAX_MODULES_PER_QR.
    let mut size = DEFAULT_CHUNK_SIZE.min(bytes.max(1));
    let min = 64usize;
    loop {
        let probe_chunk = vec![0u8; size / 2]; // conservative (real frame adds overhead)
        let frame = Frame::Chunk(VisualChunk {
            version: VERSION,
            index: 0,
            chunk_hash: [0u8; 32],
            data: probe_chunk,
        });
        let payload = pack_frame(&frame).unwrap();
        if let Ok(qr) = qr_for_bytes(&payload) {
            if qr.width() <= MAX_MODULES_PER_QR {
                return size.max(min);
            }
        }
        let next = size.saturating_sub(size / 5).max(min);
        if next == size {
            return size;
        }
        size = next;
    }
}

// Build QR codes once; decide layout & scale before rendering bitmaps.
fn frames_to_qrcodes(frames: &[Frame]) -> Result<Vec<QrCode>> {
    let mut out = Vec::with_capacity(frames.len());
    for (idx, fr) in frames.iter().enumerate() {
        let payload = pack_frame(fr)?;
        let qr = qr_for_bytes(&payload)?;
        out.push(qr);
        if idx % 25 == 0 {
            step!("Encoded {} / {} QRs…", idx + 1, frames.len());
        }
    }
    Ok(out)
}

fn estimate_cell_size(modules: u32, module_px: u32) -> (u32, u32) {
    // Total QR bitmap size with quiet zone, then add tile margins.
    let total_modules = modules + QUIET_ZONE_MODULES * 2;
    let side_px = total_modules * module_px;
    let cell_w = side_px + TILE_MARGIN_PX * 2;
    let cell_h = side_px + TILE_MARGIN_PX * 2;
    (cell_w, cell_h)
}

fn choose_layout(qr_modules_max: u32, n: u32) -> (u32, u32, u32) {
    // Pick a module_px (down to MIN_MODULE_PX) and columns that keep canvas within MAX_CANVAS_DIM.
    let mut module_px = DEFAULT_MODULE_PX;
    loop {
        let (cell_w, cell_h) = estimate_cell_size(qr_modules_max, module_px);

        // Choose columns as many as fit within MAX_CANVAS_DIM
        let max_cols = ((MAX_CANVAS_DIM.saturating_sub(GRID_GAP_PX)) / (cell_w + GRID_GAP_PX)).max(1);
        let cols = max_cols.min(n.max(1));
        let rows = ((n + cols - 1) / cols).max(1);

        let total_w = cols * cell_w + (cols + 1) * GRID_GAP_PX;
        let total_h = rows * cell_h + (rows + 1) * GRID_GAP_PX;

        if total_w <= MAX_CANVAS_DIM && total_h <= MAX_CANVAS_DIM {
            return (module_px, cols, rows);
        }

        if module_px == MIN_MODULE_PX {
            // Give up shrinking; return whatever we have (may exceed bound, but avoids infinite loop)
            return (module_px, cols, rows);
        }
        module_px -= 1;
    }
}

fn layout_and_render_grid(qrs: &[QrCode]) -> RgbaImage {
    let n = qrs.len() as u32;
    let qr_modules_max = qrs.iter().map(|q| q.width() as u32).max().unwrap_or(21);
    let (module_px, cols, rows) = choose_layout(qr_modules_max, n);

    let (cell_w, cell_h) = estimate_cell_size(qr_modules_max, module_px);
    let total_w = cols * cell_w + (cols + 1) * GRID_GAP_PX;
    let total_h = rows * cell_h + (rows + 1) * GRID_GAP_PX;

    step!(
        "Compositing QR grid… (module_px={}, cells={}×{}, canvas={}×{} px)",
        module_px,
        cols,
        rows,
        total_w,
        total_h
    );

    let mut canvas: RgbaImage = ImageBuffer::from_pixel(total_w, total_h, Rgba([255, 255, 255, 255]));

    for (i, qr) in qrs.iter().enumerate() {
        let i = i as u32;
        let cx = i % cols;
        let cy = i / cols;

        let x0 = GRID_GAP_PX + cx * (cell_w + GRID_GAP_PX);
        let y0 = GRID_GAP_PX + cy * (cell_h + GRID_GAP_PX);

        // Render bitmap for this QR with chosen module_px and quiet zone
        let bmp = render_qr_to_rgba(qr, module_px, QUIET_ZONE_MODULES);

        // Center bitmap within cell
        let off_x = x0 + (cell_w - bmp.width()) / 2;
        let off_y = y0 + (cell_h - bmp.height()) / 2;

        imageops::overlay(&mut canvas, &bmp, off_x as i64, off_y as i64);
    }
    canvas
}

// -------------------- Encode (file → visual PNG) --------------------

fn encode_file_to_visual_png(input: &Path) -> Result<PathBuf> {
    step!("Encoding file {:?} → visual PNG (QR mosaic)…", input.file_name().unwrap_or_default());

    step!("Reading & hashing input…");
    let bytes = fs::read(input).with_context(|| format!("read {:?}", input))?;
    let (file_hash, file_len_stream) = blake3_hash_file(input)?;
    if bytes.len() as u64 != file_len_stream {
        bail!("file size changed during read ({} vs {})", bytes.len(), file_len_stream);
    }
    ok!("Input: {} bytes, BLAKE3={}", bytes.len(), hex32(file_hash));

    let filename = input.file_name().and_then(OsStr::to_str).unwrap_or("unknown").to_string();

    step!("Computing chunk plan…");
    let chunk_size = choose_chunk_size(bytes.len());
    let num_chunks = (bytes.len() + chunk_size - 1) / chunk_size;
    ok!("Chunk size: {} bytes, chunks: {}", chunk_size, num_chunks);

    let header = VisualHeader {
        version: VERSION,
        filename: filename.clone(),
        total_len: bytes.len() as u64,
        file_hash,
        chunk_size: chunk_size as u32,
        num_chunks: num_chunks as u32,
    };
    let header_frame = Frame::Header(header);

    step!("Building QR frames…");
    let mut frames: Vec<Frame> = Vec::with_capacity(HEADER_REPLICATION + num_chunks);
    for _ in 0..HEADER_REPLICATION {
        if let Frame::Header(h) = &header_frame {
            frames.push(Frame::Header(h.clone()));
        }
    }
    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(bytes.len());
        let chunk = &bytes[start..end];
        let c_hash = blake3_hash_bytes(chunk);
        frames.push(Frame::Chunk(VisualChunk {
            version: VERSION,
            index: i as u32,
            chunk_hash: c_hash,
            data: chunk.to_vec(),
        }));
    }

    // Encode frames to QR codes (bit matrices), report progress.
    let qrcodes = frames_to_qrcodes(&frames)?;
    ok!("QRs encoded: {}", qrcodes.len());

    // Compose into a grid with adaptive module size to respect canvas limits.
    let canvas = layout_and_render_grid(&qrcodes);

    // Output filename
    let out = if let Some(ext) = input.extension().and_then(OsStr::to_str) {
        input.with_extension(format!("{ext}.png"))
    } else {
        input.with_extension("png")
    };

    step!("Writing PNG to {:?}…", out.file_name().unwrap_or_default());
    DynamicImage::ImageRgba8(canvas).save(&out).context("write png")?;
    ok!("PNG written.");

    // Active proof: re-open and decode; verify we recover exact bytes
    step!("Re-opening PNG for visual decode verification…");
    let (hdr2, chunks2) = visual_decode_collect(&out)?;
    let file_bytes = assemble_from_frames(&hdr2, chunks2)?;
    let verified = verify_bytes(hdr2.total_len, hdr2.file_hash, UnverifiedBytes(file_bytes))?;
    if verified.as_slice() != bytes.as_slice() {
        bail!("post-write byte-for-byte comparison failed");
    }
    ok!("Round-trip verification OK (length + BLAKE3 + exact bytes).");

    eprintln!();
    ok!("ENCODE COMPLETE → {:?}", out.file_name().unwrap_or_default());
    Ok(out)
}

// -------------------- Decode (visual image → file) --------------------

fn threshold(gray: &GrayImage, t: u8) -> GrayImage {
    let mut out = gray.clone();
    for p in out.pixels_mut() {
        p.0[0] = if p.0[0] >= t { 255 } else { 0 };
    }
    out
}

fn try_detect(gray: &GrayImage) -> (Vec<VisualHeader>, Vec<VisualChunk>) {
    let mut decoder = Quirc::default();
    let codes: Vec<_> = decoder
        .identify(gray.width() as usize, gray.height() as usize, gray.as_raw())
        .collect();

    let mut headers: Vec<VisualHeader> = Vec::new();
    let mut chunks: Vec<VisualChunk> = Vec::new();

    for code_res in codes {
        let code = match code_res {
            Ok(c) => c,
            Err(_) => continue,
        };
        let Ok(decoded) = code.decode() else { continue };
        let payload = decoded.payload; // Vec<u8>
        if let Ok(frame) = unpack_frame(&payload) {
            match frame {
                Frame::Header(h) => headers.push(h),
                Frame::Chunk(c) => chunks.push(c),
            }
        }
    }
    (headers, chunks)
}

fn visual_decode_collect(img_path: &Path) -> Result<(VisualHeader, Vec<VisualChunk>)> {
    // Load any common image format; convert to luma8.
    let dynimg = image::open(img_path).with_context(|| format!("open {:?}", img_path))?;
    let base_gray: GrayImage = dynimg.into_luma8();

    // Try original
    let (mut headers, mut chunks) = try_detect(&base_gray);
    if headers.is_empty() {
        // Try thresholded variants
        for &t in THRESHOLDS {
            step!("No headers on first pass; retrying with threshold {t} …");
            let th = threshold(&base_gray, t);
            (headers, chunks) = try_detect(&th);
            if !headers.is_empty() {
                break;
            }
        }
    }

    if headers.is_empty() {
        bail!("No header frames found.");
    }

    // Majority by identical header fields
    use std::collections::HashMap;
    fn hdr_key(h: &VisualHeader) -> (u16, String, u64, [u8; 32], u32, u32) {
        (h.version, h.filename.clone(), h.total_len, h.file_hash, h.chunk_size, h.num_chunks)
    }
    let mut counts: HashMap<(u16, String, u64, [u8; 32], u32, u32), usize> = HashMap::new();
    for h in &headers {
        *counts.entry(hdr_key(h)).or_default() += 1;
    }
    let (key, _) = counts.into_iter().max_by_key(|(_, c)| *c).unwrap();
    let header = headers.into_iter().find(|h| hdr_key(h) == key).unwrap();

    ok!(
        "Collected frames: header x{}, chunks x{}.",
        HEADER_REPLICATION, chunks.len()
    );
    Ok((header, chunks))
}

fn assemble_from_frames(header: &VisualHeader, mut chunks: Vec<VisualChunk>) -> Result<Vec<u8>> {
    // Keep only valid chunks: index in range & hash matches & unique index
    let mut by_index: Vec<Option<Vec<u8>>> = vec![None; header.num_chunks as usize];
    let mut valid = 0usize;
    for c in chunks.drain(..) {
        let idx = c.index as usize;
        if idx >= by_index.len() {
            continue;
        }
        if blake3_hash_bytes(&c.data) != c.chunk_hash {
            continue;
        }
        if by_index[idx].is_none() {
            by_index[idx] = Some(c.data);
            valid += 1;
        }
    }
    if valid < by_index.len() {
        bail!("missing chunks: have {} / {}", valid, by_index.len());
    }

    let mut out = Vec::with_capacity(header.total_len as usize);
    for (i, opt) in by_index.into_iter().enumerate() {
        let mut chunk = opt.ok_or_else(|| anyhow!("missing chunk at index {}", i))?;
        out.append(&mut chunk);
    }
    out.truncate(header.total_len as usize);
    if blake3_hash_bytes(&out) != header.file_hash {
        bail!("assembled file hash mismatch");
    }
    Ok(out)
}

fn decode_image_to_file(input: &Path) -> Result<PathBuf> {
    step!("Decoding image {:?} → original file…", input.file_name().unwrap_or_default());

    let (header, chunks) = visual_decode_collect(input)?;
    ok!(
        "Header: file {:?}, len={}, chunks={}",
        header.filename, header.total_len, header.num_chunks
    );

    let file_bytes = assemble_from_frames(&header, chunks)?;
    let verified = verify_bytes(header.total_len, header.file_hash, UnverifiedBytes(file_bytes))?;

    // Output path (idempotent behavior)
    let desired_name = if header.filename.trim().is_empty() {
        "decoded.bin".to_string()
    } else {
        header.filename.clone()
    };
    let mut desired_path = input.with_file_name(&desired_name);

    if desired_path.exists() {
        let (h, l) = blake3_hash_file(&desired_path)?;
        if l == header.total_len && h == header.file_hash {
            ok!("Existing file matches payload; nothing to write.");
            eprintln!();
            ok!("DECODE COMPLETE → {:?}", desired_path.file_name().unwrap_or_default());
            return Ok(desired_path);
        }
        warn!(
            "{:?} exists with different content; appending .restored",
            desired_path.file_name().unwrap_or_default()
        );
        desired_path = input.with_file_name(format!("{}.restored", desired_name));
    }

    step!("Writing reconstructed file to {:?}…", desired_path.file_name().unwrap_or_default());
    fs::write(&desired_path, verified.as_slice()).with_context(|| format!("write {:?}", desired_path))?;
    ok!("File written.");

    // Re-verify from disk
    step!("Re-reading written file to re-verify…");
    let (rehash, relen) = blake3_hash_file(&desired_path)?;
    if relen != header.total_len {
        bail!("output length mismatch after write ({} vs {})", relen, header.total_len);
    }
    if rehash != header.file_hash {
        bail!("output BLAKE3 mismatch after write");
    }
    ok!("Output verified (length + BLAKE3).");

    eprintln!();
    ok!("DECODE COMPLETE → {:?}", desired_path.file_name().unwrap_or_default());
    Ok(desired_path)
}

// -------------------- Entry: one-arg tool --------------------

fn main() {
    if let Err(e) = real_main() {
        fail!("{e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let Some(path_os) = args.next() else {
        eprintln!("Usage: pixpack <path-to-file-or-image>");
        bail!("missing required argument");
    };
    if args.next().is_some() {
        bail!("exactly one argument is required");
    }
    let path = PathBuf::from(path_os);
    if !path.exists() {
        bail!("path does not exist: {:?}", path);
    }

    // If it looks like an image, try to decode visually first.
    if image::open(&path).is_ok() {
        if let Ok(_out) = decode_image_to_file(&path) {
            return Ok(());
        } else {
            step!("Input is an image but not a pixpack QR mosaic; proceeding to encode it as data.");
        }
    }

    // Encode anything else (including images without pixpack payload) into a new PNG QR mosaic.
    encode_file_to_visual_png(&path)?;
    Ok(())
}

// -------------------- Tests --------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // Simple deterministic xorshift64 PRNG to generate pseudo-random test data.
    fn xorshift64(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
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
    fn frame_header_roundtrip() -> Result<()> {
        let h = VisualHeader {
            version: super::VERSION,
            filename: "x.bin".into(),
            total_len: 123,
            file_hash: [1u8; 32],
            chunk_size: 256,
            num_chunks: 2,
        };
        let fr = super::Frame::Header(h);
        let b = super::pack_frame(&fr)?;
        let fr2 = super::unpack_frame(&b)?;
        match fr2 {
            super::Frame::Header(h2) => {
                assert_eq!(h2.filename, "x.bin");
                assert_eq!(h2.total_len, 123);
                assert_eq!(h2.chunk_size, 256);
                assert_eq!(h2.num_chunks, 2);
            }
            _ => panic!("wrong frame kind"),
        }
        Ok(())
    }

    #[test]
    fn frame_chunk_roundtrip() -> Result<()> {
        let c = VisualChunk {
            version: super::VERSION,
            index: 7,
            chunk_hash: [2u8; 32],
            data: b"hello".to_vec(),
        };
        let fr = super::Frame::Chunk(c);
        let b = super::pack_frame(&fr)?;
        let fr2 = super::unpack_frame(&b)?;
        match fr2 {
            super::Frame::Chunk(c2) => {
                assert_eq!(c2.index, 7);
                assert_eq!(&c2.data, b"hello");
            }
            _ => panic!("wrong frame kind"),
        }
        Ok(())
    }

    #[test]
    fn choose_chunk_size_is_reasonable() {
        let s = super::choose_chunk_size(50_000);
        assert!(s >= 64 && s <= 2000, "chunk size chosen: {}", s);
    }

    // ---------- Integration-style tests (touch disk) ----------

    #[test]
    fn roundtrip_small_text() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("note.txt");
        fs::write(&input, b"Hello visual QR world!\n")?;
        let png = super::encode_file_to_visual_png(&input)?;
        let out = super::decode_image_to_file(&png)?;
        assert_eq!(fs::read(&input)?, fs::read(&out)?);
        Ok(())
    }

    #[test]
    fn roundtrip_large_binary() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("blob.bin");
        let data = gen_bytes(300 * 1024, 0xFEEDFACE);
        fs::write(&input, &data)?;
        let png = super::encode_file_to_visual_png(&input)?;
        let out = super::decode_image_to_file(&png)?;
        assert_eq!(blake3_hash_bytes(&data), blake3_hash_file(&out)?.0);
        Ok(())
    }

    #[test]
    fn decode_idempotent_when_same_file_exists() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("artifact.bin");
        fs::write(&input, b"XYZ")?;
        let png = super::encode_file_to_visual_png(&input)?;
        let out1 = super::decode_image_to_file(&png)?;
        assert_eq!(out1.file_name().unwrap().to_str().unwrap(), "artifact.bin");
        let out2 = super::decode_image_to_file(&png)?;
        assert_eq!(out2, out1);
        Ok(())
    }

    #[test]
    fn screenshot_like_rescale_still_decodes() -> Result<()> {
        // Simulate a screenshot: rescale by 200% and re-save as PNG
        let dir = tempdir()?;
        let input = dir.path().join("doc.txt");
        fs::write(&input, b"Pixels survive screenshots!")?;
        let png = super::encode_file_to_visual_png(&input)?;
        let img = image::open(&png)?.to_rgba8();
        let big = imageops::resize(&img, img.width() * 2, img.height() * 2, imageops::FilterType::Nearest);
        let big_path = png.with_file_name("screenshot.png");
        DynamicImage::ImageRgba8(big).save(&big_path)?;
        let out = super::decode_image_to_file(&big_path)?;
        assert_eq!(fs::read(&input)?, fs::read(&out)?);
        Ok(())
    }

    #[test]
    fn filename_conflict_restored_when_different() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("report.dat");
        fs::write(&input, b"ORIGINAL")?;
        let png = super::encode_file_to_visual_png(&input)?;

        // Create a conflicting file with different content
        let conflict = png.with_file_name("report.dat");
        fs::write(&conflict, b"DIFFERENT")?;

        let out = super::decode_image_to_file(&png)?;
        assert!(out.file_name().unwrap().to_str().unwrap().ends_with(".restored"));
        assert_eq!(fs::read(out)?, b"ORIGINAL");
        Ok(())
    }

    #[test]
    fn tamper_detection_hash_mismatch() -> Result<()> {
        // Decode, corrupt one chunk frame re-encoded into a *new* QR mosaic, then fail.
        let dir = tempdir()?;
        let input = dir.path().join("secret.bin");
        fs::write(&input, b"TOP_SECRET_BYTES")?;
        let png = super::encode_file_to_visual_png(&input)?;

        // Load & decode frames
        let (hdr, mut chunks) = super::visual_decode_collect(&png)?;
        // Corrupt first chunk if any
        if let Some(first) = chunks.get_mut(0) {
            if !first.data.is_empty() {
                first.data[0] ^= 1;
                first.chunk_hash = blake3_hash_bytes(&first.data);
            }
        }
        // Forge a new image with the (possibly corrupted) frames
        let mut frames: Vec<super::Frame> = Vec::new();
        for _ in 0..super::HEADER_REPLICATION {
            frames.push(super::Frame::Header(hdr.clone()));
        }
        for c in &chunks {
            frames.push(super::Frame::Chunk(super::VisualChunk {
                version: super::VERSION,
                index: c.index,
                chunk_hash: c.chunk_hash,
                data: c.data.clone(),
            }));
        }
        let qrs = super::frames_to_qrcodes(&frames)?;
        let canvas = super::layout_and_render_grid(&qrs);

        let forged = png.with_file_name("forged.png");
        DynamicImage::ImageRgba8(canvas).save(&forged)?;

        // Now decoding should fail (hash mismatch or assembly problems)
        let err = super::decode_image_to_file(&forged).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("hash mismatch") ||
            msg.contains("missing chunks") ||
            msg.contains("assembled file hash mismatch"),
            "unexpected error: {msg}"
        );
        Ok(())
    }
}
