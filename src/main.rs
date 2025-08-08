// src/main.rs
//
// Single-arg CLI:
//   - If the given path visually decodes as a pixpack page (any common image format): DECODE the
//     whole dataset from all sibling pages in the same directory to the original file (with proofs).
//   - Otherwise: ENCODE the file into one or more *viewable* PNG pages of QR tiles (with proofs).
//
// Integrity framing per QR (versioned):
//   magic:   b"PXP2"
//   kind:    0x01 = Header | 0x02 = Chunk
//   body:    postcard-serialized struct (VisualHeader/VisualChunk)
//
// VisualHeader {
//   version, dataset_id[16],
//   filename, total_len, file_hash[32],
//   chunk_size, num_chunks,
//   page_no, page_count, first_chunk_index, chunks_in_page
// }
// VisualChunk { version, index, chunk_hash[32], data }
//
// Active proofs:
//   - Before encode: hash whole input (BLAKE3). After writing *all* pages, re-decode the dataset
//     from disk (scanning sibling pages) and byte-compare + length & BLAKE3.
//   - On decode: verify each chunk hash and the final whole-file BLAKE3 and length.
//   - Safe idempotency: if decoded file already exists and matches, we don't overwrite.
//
// Build: cargo build --release
// Run:   cargo run --release -- <path>
//
// Requires Cargo.toml deps (regular): anyhow, blake3, image="0.25", qrcode="0.14.1", quircs="0.10.2",
// rayon, serde(derive), postcard(alloc,use-std). dev-deps: tempfile.

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
use qrcode::{types::Color, EcLevel, QrCode};
use quircs::Quirc;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// -------------------- Constants & types --------------------

const MAGIC: &[u8; 4] = b"PXP2";
const VERSION: u16 = 2;

// Visual/layout parameters
const DEFAULT_CHUNK_SIZE: usize = 4096;  // Start big to reduce QR count
const MIN_CHUNK_SIZE: usize = 256;       // Lower bound when trying to fit
const MAX_MODULES_PER_QR: usize = 105;   // max QR version width we'll accept when planning
const DEFAULT_MODULE_PX: u32 = 10;       // initial pixels per module
const MIN_MODULE_PX: u32 = 4;            // minimum pixels per module to keep detection robust
const TILE_MARGIN_PX: u32 = 18;          // padding around each QR tile
const GRID_GAP_PX: u32 = 16;             // spacing between tiles
const QUIET_ZONE_MODULES: u32 = 4;       // additional quiet zone around QR content
const HEADER_REPLICATION: usize = 3;     // repeat header QR for robustness
const MAX_CANVAS_DIM: u32 = 8000;        // bound final PNG width/height

// Decode fallbacks (binarization thresholds)
const THRESHOLDS: &[u8] = &[190, 170, 150, 130, 110, 96];

#[derive(Debug, Clone)]
struct UnverifiedBytes(Vec<u8>);
#[derive(Debug, Clone)]
struct VerifiedBytes(Vec<u8>);
impl VerifiedBytes {
    fn as_slice(&self) -> &[u8] { &self.0 }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct VisualHeader {
    version: u16,
    dataset_id: [u8; 16],
    filename: String,
    total_len: u64,
    file_hash: [u8; 32],
    chunk_size: u32,
    num_chunks: u32,
    page_no: u32,
    page_count: u32,
    first_chunk_index: u32,
    chunks_in_page: u32,
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
macro_rules! step { ($($arg:tt)*) => { eprintln!("▶ {}", format!($($arg)*)); }; }
macro_rules! ok   { ($($arg:tt)*) => { eprintln!("✔ {}", format!($($arg)*)); }; }
macro_rules! warn { ($($arg:tt)*) => { eprintln!("⚠ {}", format!($($arg)*)); }; }
macro_rules! fail { ($($arg:tt)*) => { eprintln!("✘ {}", format!($($arg)*)); }; }

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
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

// Deterministic dataset id (16 bytes) derived from file content + filename + len + timestamp
fn derive_dataset_id(bytes: &[u8], filename: &str) -> [u8; 16] {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.update(filename.as_bytes());
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    hasher.update(&ts.to_le_bytes());
    let digest = hasher.finalize();
    let mut id = [0u8; 16];
    id.copy_from_slice(&digest.as_bytes()[..16]);
    id
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
    if payload.len() < MAGIC.len() + 1 { bail!("QR payload too short"); }
    if &payload[..4] != MAGIC { bail!("QR magic mismatch"); }
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
    QrCode::with_error_correction_level(data, EcLevel::H)
        .map_err(|e| anyhow!("QR encode failed: {e}"))
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

fn estimate_cell_size(modules: u32, module_px: u32) -> (u32, u32) {
    let total_modules = modules + QUIET_ZONE_MODULES * 2;
    let side_px = total_modules * module_px;
    let cell_w = side_px + TILE_MARGIN_PX * 2;
    let cell_h = side_px + TILE_MARGIN_PX * 2;
    (cell_w, cell_h)
}

fn plan_grid_capacity(qr_modules_max: u32, mut module_px: u32) -> (u32, u32, u32) {
    // Returns (module_px, cols, rows) that fit within MAX_CANVAS_DIM,
    // with module_px >= MIN_MODULE_PX, maximizing module size.
    loop {
        let (cell_w, cell_h) = estimate_cell_size(qr_modules_max, module_px);
        let max_cols = ((MAX_CANVAS_DIM.saturating_sub(GRID_GAP_PX)) / (cell_w + GRID_GAP_PX)).max(1);
        let max_rows = ((MAX_CANVAS_DIM.saturating_sub(GRID_GAP_PX)) / (cell_h + GRID_GAP_PX)).max(1);
        if max_cols > 0 && max_rows > 0 {
            return (module_px, max_cols, max_rows);
        }
        if module_px == MIN_MODULE_PX { return (module_px, 1, 1); }
        module_px -= 1;
    }
}

fn worst_qr_modules_for(chunk_size: usize, header_example: &VisualHeader) -> Result<(u32, u32)> {
    // Returns (header_modules, chunk_modules)
    let header_payload = pack_frame(&Frame::Header(header_example.clone()))?;
    let header_qr = qr_for_bytes(&header_payload)?;
    let header_modules = header_qr.width() as u32;

    let probe_chunk = vec![0u8; chunk_size.min(8192)];
    let probe = VisualChunk {
        version: VERSION,
        index: 0,
        chunk_hash: blake3_hash_bytes(&probe_chunk),
        data: probe_chunk,
    };
    let chunk_payload = pack_frame(&Frame::Chunk(probe))?;
    let chunk_qr = qr_for_bytes(&chunk_payload)?;
    let chunk_modules = chunk_qr.width() as u32;

    Ok((header_modules, chunk_modules))
}

// Choose a chunk size that yields QR width <= MAX_MODULES_PER_QR and allows at least
// HEADER_REPLICATION + 1 frames per page with module_px >= MIN_MODULE_PX.
fn choose_plan(bytes: usize, header_example: &VisualHeader) -> Result<(usize, u32, u32, u32)> {
    // returns (chunk_size, module_px, cols, rows)
    let mut chunk_size = DEFAULT_CHUNK_SIZE.min(bytes.max(1));
    let min_chunk = MIN_CHUNK_SIZE.min(bytes.max(1));
    loop {
        let (_, chunk_modules) = worst_qr_modules_for(chunk_size, header_example)?;
        if chunk_modules as usize > MAX_MODULES_PER_QR {
            // QR version too large; reduce chunk size
            let next = chunk_size.saturating_sub(chunk_size / 4).max(min_chunk);
            if next == chunk_size { /* cannot reduce further */ }
            chunk_size = next;
        }
        // Pick module size and grid capacity
        let (module_px, cols, rows) = plan_grid_capacity(chunk_modules, DEFAULT_MODULE_PX);
        let per_page_capacity = cols * rows;
        if per_page_capacity >= (HEADER_REPLICATION as u32 + 1) {
            return Ok((chunk_size.max(min_chunk), module_px, cols, rows));
        }
        // Not enough tiles per page even for 1 chunk; shrink chunk to reduce QR modules
        if chunk_size == min_chunk && module_px == MIN_MODULE_PX {
            // as a last resort, still accept plan (very small capacity)
            return Ok((chunk_size, module_px, cols, rows));
        }
        let next = chunk_size.saturating_sub(chunk_size / 4).max(min_chunk);
        if next == chunk_size {
            // Can't reduce further; accept plan
            return Ok((chunk_size, module_px, cols, rows));
        }
        chunk_size = next;
    }
}

// Render a page from frames with known module_px and grid (cols, rows).
fn render_page(frames: &[Frame], module_px: u32, cols: u32, rows: u32) -> Result<RgbaImage> {
    // Build payloads + QRs in parallel
    let payloads: Vec<Vec<u8>> = frames.par_iter()
        .map(|fr| pack_frame(fr))
        .collect::<Result<Vec<_>>>()?;
    let qrs: Vec<QrCode> = payloads.par_iter()
        .map(|b| qr_for_bytes(b))
        .collect::<Result<Vec<_>>>()?;

    // Cell size derived from max modules among frames
    let qr_modules_max = qrs.iter().map(|q| q.width() as u32).max().unwrap_or(21);
    let (cell_w, cell_h) = estimate_cell_size(qr_modules_max, module_px);
    let total_w = cols * cell_w + (cols + 1) * GRID_GAP_PX;
    let total_h = rows * cell_h + (rows + 1) * GRID_GAP_PX;

    step!(
        "Compositing page… (module_px={}, cells={}×{}, canvas={}×{} px)",
        module_px, cols, rows, total_w, total_h
    );

    // Pre-render bitmaps in parallel
    let bitmaps: Vec<RgbaImage> = qrs.par_iter()
        .map(|qr| render_qr_to_rgba(qr, module_px, QUIET_ZONE_MODULES))
        .collect();

    let mut canvas: RgbaImage = ImageBuffer::from_pixel(total_w, total_h, Rgba([255, 255, 255, 255]));

    for (i, bmp) in bitmaps.into_iter().enumerate() {
        let i = i as u32;
        let cx = i % cols;
        let cy = i / cols;

        let x0 = GRID_GAP_PX + cx * (cell_w + GRID_GAP_PX);
        let y0 = GRID_GAP_PX + cy * (cell_h + GRID_GAP_PX);

        // Center inside cell (bitmaps can vary by a few pixels between header/chunk versions)
        let off_x = x0 + (cell_w - bmp.width()) / 2;
        let off_y = y0 + (cell_h - bmp.height()) / 2;
        imageops::overlay(&mut canvas, &bmp, off_x as i64, off_y as i64);
    }

    Ok(canvas)
}

// -------------------- Encode (file → visual PNG pages) --------------------

struct PagePlan {
    module_px: u32,
    cols: u32,
    rows: u32,
    frames_per_page: u32,
    chunk_size: usize,
    num_chunks: usize,
}

fn build_page_plan(total_bytes: usize, filename: &str, file_hash: [u8; 32], dataset_id: [u8; 16]) -> Result<PagePlan> {
    // Header example for sizing (page fields are placeholders here)
    let header_example = VisualHeader {
        version: VERSION,
        dataset_id,
        filename: filename.to_string(),
        total_len: total_bytes as u64,
        file_hash,
        chunk_size: DEFAULT_CHUNK_SIZE as u32,
        num_chunks: 0,
        page_no: 1,
        page_count: 1,
        first_chunk_index: 0,
        chunks_in_page: 1,
    };
    let (chunk_size, module_px, cols, rows) = choose_plan(total_bytes, &header_example)?;
    let frames_per_page = cols * rows;
    let num_chunks = (total_bytes + chunk_size - 1) / chunk_size;
    Ok(PagePlan {
        module_px,
        cols,
        rows,
        frames_per_page,
        chunk_size,
        num_chunks,
    })
}

fn encode_file_to_visual_pngs(input: &Path) -> Result<Vec<PathBuf>> {
    step!("Encoding file {:?} → visual PNG pages…", input.file_name().unwrap_or_default());

    step!("Reading & hashing input…");
    let bytes = fs::read(input).with_context(|| format!("read {:?}", input))?;
    let (file_hash_stream, file_len_stream) = blake3_hash_file(input)?;
    if bytes.len() as u64 != file_len_stream {
        bail!("file size changed during read ({} vs {})", bytes.len(), file_len_stream);
    }
    let file_hash = blake3_hash_bytes(&bytes);
    if file_hash != file_hash_stream {
        bail!("file content changed between reads (hash mismatch)");
    }
    ok!("Input: {} bytes, BLAKE3={}", bytes.len(), hex32(file_hash));

    let filename = input.file_name().and_then(OsStr::to_str).unwrap_or("unknown").to_string();
    let dataset_id = derive_dataset_id(&bytes, &filename);

    step!("Planning pages & chunks…");
    let plan = build_page_plan(bytes.len(), &filename, file_hash, dataset_id)?;
    let per_page_capacity = plan.frames_per_page as usize;
    let header_overhead = HEADER_REPLICATION;
    let chunks_per_page = per_page_capacity.saturating_sub(header_overhead).max(1);
    let page_count = (plan.num_chunks + chunks_per_page - 1) / chunks_per_page;
    ok!(
        "chunk_size={} bytes, num_chunks={}, module_px={}, grid={}×{}, frames/page={}, pages={}",
        plan.chunk_size, plan.num_chunks, plan.module_px, plan.cols, plan.rows, per_page_capacity, page_count
    );

    // Prepare output filenames
    let mut outputs = Vec::with_capacity(page_count);
    let stem = input.file_stem().and_then(OsStr::to_str).unwrap_or("file");
    let (base, extpart) = if let Some(ext) = input.extension().and_then(OsStr::to_str) {
        (format!("{stem}.{ext}"), ext.to_string())
    } else {
        (stem.to_string(), String::from(""))
    };

    // Build all pages
    for p in 0..page_count {
        step!("Building page {}/{}…", p + 1, page_count);
        let first_chunk_index = p * chunks_per_page;
        let last = ((p + 1) * chunks_per_page).min(plan.num_chunks);
        let this_count = last - first_chunk_index;

        // Page header
        let header = VisualHeader {
            version: VERSION,
            dataset_id,
            filename: filename.clone(),
            total_len: bytes.len() as u64,
            file_hash,
            chunk_size: plan.chunk_size as u32,
            num_chunks: plan.num_chunks as u32,
            page_no: (p + 1) as u32,
            page_count: page_count as u32,
            first_chunk_index: first_chunk_index as u32,
            chunks_in_page: this_count as u32,
        };

        // Frames for this page
        let mut frames: Vec<Frame> = Vec::with_capacity(header_overhead + this_count);
        for _ in 0..HEADER_REPLICATION { frames.push(Frame::Header(header.clone())); }
        for i in 0..this_count {
            let idx = first_chunk_index + i;
            let start = idx * plan.chunk_size;
            let end = (start + plan.chunk_size).min(bytes.len());
            let chunk = &bytes[start..end];
            frames.push(Frame::Chunk(VisualChunk {
                version: VERSION,
                index: idx as u32,
                chunk_hash: blake3_hash_bytes(chunk),
                data: chunk.to_vec(),
            }));
        }

        // Render page
        let canvas = render_page(&frames, plan.module_px, plan.cols, plan.rows)?;

        // Output path
        let out = if page_count == 1 {
            if !extpart.is_empty() { input.with_extension(format!("{extpart}.png")) }
            else { input.with_extension("png") }
        } else {
            // multi-page: base + .pxpNNN.png
            let name = format!("{base}.pxp{:03}.png", p + 1);
            input.with_file_name(name)
        };

        step!("Writing page to {:?}…", out.file_name().unwrap_or_default());
        DynamicImage::ImageRgba8(canvas).save(&out).context("write png")?;
        ok!("Page written.");
        outputs.push(out);
    }

    // Active proof: re-decode whole dataset from disk (scan directory), verify bytes
    step!("Re-decoding dataset from pages to verify…");
    let (hdr, chunks) = decode_dataset_from_any_page(&outputs[0])?;
    let rebuilt = assemble_from_frames_all_pages(&hdr, chunks)?;
    let verified = verify_bytes(hdr.total_len, hdr.file_hash, UnverifiedBytes(rebuilt))?;
    if verified.as_slice() != bytes.as_slice() {
        bail!("post-write byte-for-byte comparison failed");
    }
    ok!("Round-trip verification OK (length + BLAKE3 + exact bytes).");

    eprintln!();
    ok!("ENCODE COMPLETE → {} page(s)", outputs.len());
    Ok(outputs)
}

// -------------------- Decode (visual image(s) → file) --------------------

fn threshold(gray: &GrayImage, t: u8) -> GrayImage {
    let mut out = gray.clone();
    for p in out.pixels_mut() {
        p.0[0] = if p.0[0] >= t { 255 } else { 0 };
    }
    out
}

#[derive(Debug, Clone)]
struct PageFrames {
    header: VisualHeader,
    chunks: Vec<VisualChunk>,
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
        let payload = decoded.payload;
        if let Ok(frame) = unpack_frame(&payload) {
            match frame {
                Frame::Header(h) => headers.push(h),
                Frame::Chunk(c) => chunks.push(c),
            }
        }
    }
    (headers, chunks)
}

fn collect_from_image(img_path: &Path) -> Result<PageFrames> {
    let dynimg = image::open(img_path).with_context(|| format!("open {:?}", img_path))?;
    let base_gray: GrayImage = dynimg.into_luma8();

    // First pass
    let (mut headers, mut chunks) = try_detect(&base_gray);
    // Try thresholds
    if headers.is_empty() {
        for &t in THRESHOLDS {
            step!("No headers in {:?}; retry threshold {t}…", img_path.file_name().unwrap_or_default());
            let th = threshold(&base_gray, t);
            (headers, chunks) = try_detect(&th);
            if !headers.is_empty() { break; }
        }
    }

    if headers.is_empty() { bail!("No header frames found."); }

    // Majority match
    use std::collections::HashMap;
    fn hdr_key(h: &VisualHeader) -> (u16, [u8;16], String, u64, [u8;32], u32, u32, u32, u32) {
        (h.version, h.dataset_id, h.filename.clone(), h.total_len, h.file_hash, h.chunk_size, h.num_chunks, h.page_count, h.chunks_in_page)
    }
    let mut counts: HashMap<(u16,[u8;16],String,u64,[u8;32],u32,u32,u32,u32), usize> = HashMap::new();
    for h in &headers { *counts.entry(hdr_key(h)).or_default() += 1; }
    let (key, _) = counts.into_iter().max_by_key(|(_, c)| *c).unwrap();
    let header = headers.into_iter().find(|h| hdr_key(h) == key).unwrap();

    Ok(PageFrames { header, chunks })
}

// Assemble by combining chunks from ALL pages with same dataset_id in the directory.
fn decode_dataset_from_any_page(any_page: &Path) -> Result<(VisualHeader, Vec<VisualChunk>)> {
    let first = collect_from_image(any_page)?;
    let dir = any_page.parent().unwrap_or(Path::new(".")).to_path_buf();
    let dataset_id = first.header.dataset_id;
    let hash = first.header.file_hash;
    let version = first.header.version;

    // Gather all PNGs in the same dir and decode in parallel
    let mut candidates: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let p = entry.path();
        if let Some(ext) = p.extension().and_then(OsStr::to_str) {
            if ext.eq_ignore_ascii_case("png") {
                candidates.push(p);
            }
        }
    }

    let mut pages: Vec<PageFrames> = candidates.par_iter()
        .filter_map(|p| {
            match collect_from_image(p) {
                Ok(fr) if fr.header.version == version
                    && fr.header.dataset_id == dataset_id
                    && fr.header.file_hash == hash => Some(fr),
                _ => None,
            }
        })
        .collect();

    // Ensure the current page is included (in case it wasn't matched due to naming)
    pages.push(first);

    // Deduplicate by page_no (keep the one with most chunks collected)
    use std::collections::HashMap;
    let mut best_by_page: HashMap<u32, PageFrames> = HashMap::new();
    for p in pages {
        let n = p.chunks.len() as u32;
        best_by_page
            .entry(p.header.page_no)
            .and_modify(|existing| {
                if n > existing.chunks.len() as u32 { *existing = p.clone(); }
            })
            .or_insert(p);
    }

    // Collect headers and chunks
    let mut all_chunks: Vec<VisualChunk> = Vec::new();
    let mut header_ref = None::<VisualHeader>;
    for (_, pf) in best_by_page.iter() {
        if header_ref.is_none() { header_ref = Some(pf.header.clone()); }
        all_chunks.extend_from_slice(&pf.chunks);
    }
    let header = header_ref.ok_or_else(|| anyhow!("no matching pages found in directory"))?;

    ok!(
        "Collected pages: {} / {} (chunks gathered: {})",
        best_by_page.len(),
        header.page_count,
        all_chunks.len()
    );
    Ok((header, all_chunks))
}

fn assemble_from_frames_all_pages(header: &VisualHeader, mut chunks: Vec<VisualChunk>) -> Result<Vec<u8>> {
    // Validate and merge by index
    let mut by_index: Vec<Option<Vec<u8>>> = vec![None; header.num_chunks as usize];
    let mut valid = 0usize;
    for c in chunks.drain(..) {
        let idx = c.index as usize;
        if idx >= by_index.len() { continue; }
        if blake3_hash_bytes(&c.data) != c.chunk_hash { continue; }
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

fn decode_image_to_file(entry_page: &Path) -> Result<PathBuf> {
    step!("Decoding image {:?} → original file…", entry_page.file_name().unwrap_or_default());

    let (header, chunks) = decode_dataset_from_any_page(entry_page)?;
    ok!(
        "Header: file {:?}, len={}, chunks={}, pages={}",
        header.filename, header.total_len, header.num_chunks, header.page_count
    );

    let file_bytes = assemble_from_frames_all_pages(&header, chunks)?;
    let verified = verify_bytes(header.total_len, header.file_hash, UnverifiedBytes(file_bytes))?;

    // Output path (idempotent behavior)
    let desired_name = if header.filename.trim().is_empty() { "decoded.bin".to_string() } else { header.filename.clone() };
    let mut desired_path = entry_page.with_file_name(&desired_name);

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
        desired_path = entry_page.with_file_name(format!("{}.restored", desired_name));
    }

    step!("Writing reconstructed file to {:?}…", desired_path.file_name().unwrap_or_default());
    fs::write(&desired_path, verified.as_slice()).with_context(|| format!("write {:?}", desired_path))?;
    ok!("File written.");

    // Re-verify from disk
    step!("Re-reading written file to re-verify…");
    let (rehash, relen) = blake3_hash_file(&desired_path)?;
    if relen != header.total_len { bail!("output length mismatch after write ({} vs {})", relen, header.total_len); }
    if rehash != header.file_hash { bail!("output BLAKE3 mismatch after write"); }
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
    if args.next().is_some() { bail!("exactly one argument is required"); }
    let path = PathBuf::from(path_os);
    if !path.exists() { bail!("path does not exist: {:?}", path); }

    // If it looks like an image, try to decode visually first.
    if image::open(&path).is_ok() {
        if let Ok(_out) = decode_image_to_file(&path) {
            return Ok(());
        } else {
            step!("Input is an image but not a pixpack dataset page; proceeding to encode it as data.");
        }
    }

    // Encode anything else (including images without pixpack payload) into new PNG page(s).
    let _pages = encode_file_to_visual_pngs(&path)?;
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
            dataset_id: [7u8;16],
            filename: "x.bin".into(),
            total_len: 123,
            file_hash: [1u8; 32],
            chunk_size: 256,
            num_chunks: 2,
            page_no: 1,
            page_count: 2,
            first_chunk_index: 0,
            chunks_in_page: 1,
        };
        let fr = super::Frame::Header(h.clone());
        let b = super::pack_frame(&fr)?;
        let fr2 = super::unpack_frame(&b)?;
        match fr2 {
            super::Frame::Header(h2) => {
                assert_eq!(h2.filename, "x.bin");
                assert_eq!(h2.total_len, 123);
                assert_eq!(h2.chunk_size, 256);
                assert_eq!(h2.num_chunks, 2);
                assert_eq!(h2.page_count, 2);
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
    fn choose_plan_reasonable() -> Result<()> {
        let hdr = VisualHeader {
            version: super::VERSION,
            dataset_id: [3u8;16],
            filename: "probe.bin".into(),
            total_len: 500_000,
            file_hash: [9u8; 32],
            chunk_size: super::DEFAULT_CHUNK_SIZE as u32,
            num_chunks: 0,
            page_no: 1,
            page_count: 1,
            first_chunk_index: 0,
            chunks_in_page: 1,
        };
        let (chunk_size, module_px, cols, rows) = super::choose_plan(500_000, &hdr)?;
        assert!(chunk_size >= super::MIN_CHUNK_SIZE);
        assert!(module_px >= super::MIN_MODULE_PX);
        assert!(cols >= 1 && rows >= 1);
        Ok(())
    }

    // ---------- Integration-style tests ----------

    #[test]
    fn roundtrip_small_text() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("note.txt");
        fs::write(&input, b"Hello visual QR world!\n")?;
        let pages = super::encode_file_to_visual_pngs(&input)?;
        assert!(!pages.is_empty());
        let out = super::decode_image_to_file(&pages[0])?;
        assert_eq!(fs::read(&input)?, fs::read(&out)?);
        Ok(())
    }

    #[test]
    fn screenshot_like_rescale_still_decodes() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("doc.txt");
        fs::write(&input, b"Pixels survive screenshots!")?;
        let pages = super::encode_file_to_visual_pngs(&input)?;
        let img = image::open(&pages[0])?.to_rgba8();
        let big = imageops::resize(&img, img.width() * 2, img.height() * 2, imageops::FilterType::Nearest);
        let big_path = pages[0].with_file_name("screenshot.png");
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
        let pages = super::encode_file_to_visual_pngs(&input)?;

        // Create a conflicting file with different content
        let conflict = pages[0].with_file_name("report.dat");
        fs::write(&conflict, b"DIFFERENT")?;

        let out = super::decode_image_to_file(&pages[0])?;
        assert!(out.file_name().unwrap().to_str().unwrap().ends_with(".restored"));
        assert_eq!(fs::read(out)?, b"ORIGINAL");
        Ok(())
    }

    #[test]
    fn decode_idempotent_when_same_file_exists() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("artifact.bin");
        fs::write(&input, b"XYZ")?;
        let pages = super::encode_file_to_visual_pngs(&input)?;
        let out1 = super::decode_image_to_file(&pages[0])?;
        assert_eq!(out1.file_name().unwrap().to_str().unwrap(), "artifact.bin");
        let out2 = super::decode_image_to_file(&pages[0])?;
        assert_eq!(out2, out1);
        Ok(())
    }

    #[test]
    fn roundtrip_large_binary_multipage() -> Result<()> {
        // ~1.5 MiB ensures multiple pages but keeps test time reasonable with parallelism.
        let dir = tempdir()?;
        let input = dir.path().join("bigblob.bin");
        let data = gen_bytes(1_572_864, 0xFEED_FACE_CAFE_BEEFu64);
        fs::write(&input, &data)?;
        let pages = super::encode_file_to_visual_pngs(&input)?;
        assert!(pages.len() >= 2, "expected multiple pages, got {}", pages.len());
        let out = super::decode_image_to_file(&pages[0])?;
        assert_eq!(blake3_hash_bytes(&data), blake3_hash_file(&out)?.0);
        Ok(())
    }

    #[test]
    fn tamper_detection_hash_mismatch() -> Result<()> {
        // Encode, then tamper one page's first chunk by re-rendering it with a flipped bit.
        let dir = tempdir()?;
        let input = dir.path().join("FILE.bin");
        fs::write(&input, b"TOP_FILE_BYTES_TOP_FILE_BYTES_TOP_FILE_BYTES")?;
        let pages = super::encode_file_to_visual_pngs(&input)?;
        let entry = &pages[0];

        // Load frames from first page
        let pf = super::collect_from_image(entry)?;
        let hdr = pf.header.clone();
        let mut chunks = pf.chunks.clone();
        if let Some(first) = chunks.get_mut(0) {
            if !first.data.is_empty() {
                first.data[0] ^= 1;
                first.chunk_hash = blake3_hash_bytes(&first.data);
            }
        }

        // Build a forged page with tampered chunks for page_no
        let mut frames: Vec<super::Frame> = Vec::new();
        for _ in 0..super::HEADER_REPLICATION { frames.push(super::Frame::Header(hdr.clone())); }
        for c in &chunks {
            frames.push(super::Frame::Chunk(super::VisualChunk {
                version: super::VERSION,
                index: c.index,
                chunk_hash: c.chunk_hash,
                data: c.data.clone(),
            }));
        }

        // Render with safe defaults
        let canvas = super::render_page(&frames, 6, 10, 10)?; // arbitrary grid just to produce a page
        let forged = entry.with_file_name("forged.png");
        DynamicImage::ImageRgba8(canvas).save(&forged)?;

        // Decoding entire dataset should fail (hash mismatch or missing chunks)
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
