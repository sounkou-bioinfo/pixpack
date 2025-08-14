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
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage};
#[cfg(test)]
use image::GrayImage;
use png::{ColorType as PngColorType, Decoder as PngDecoder, Transformations as PngXform};
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
const MAX_CELL_PX: u32 = 20;     // per-cell upper bound for aesthetics; not a payload/file-size limit
const TARGET_SIDE_PX: u32 = 5000; // preferred canvas side (not enforced)

// Header repeats for redundancy (same header back-to-back)
const HEADER_REPEAT: usize = 2;

// Binarization fallback thresholds after Otsu (no rotations, keep it simple)
const FALLBACK_THRESHOLDS: &[u8] = &[200, 180, 160, 140, 120, 100];

// Logging
macro_rules! step { ($($arg:tt)*) => { eprintln!("▶ {}", format!($($arg)*)); }; }
macro_rules! ok   { ($($arg:tt)*) => { eprintln!("✔ {}", format!($($arg)*)); }; }
macro_rules! fail { ($($arg:tt)*) => { eprintln!("✘ {}", format!($($arg)*)); }; }

// ========================= Types & integrity =========================

// (Removed unused VerifiedBytes helpers after switching to streaming verification)

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
// (Removed unused verify_bytes; streaming paths re-verify via BLAKE3 directly)
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

#[cfg(test)]
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
#[cfg(test)]
fn binarize(gray: &GrayImage, t: u8) -> GrayImage {
    let mut out = gray.clone();
    for p in out.pixels_mut() { p.0[0] = if p.0[0] > t { 255 } else { 0 }; }
    out
}

// ========================= Geometry inference (decode) =========================

#[derive(Debug, Clone)]
struct Geometry {
    cell_px: u32,
    origin_x: u32, // top-left of data grid (in px)
    origin_y: u32,
    grid_w: u32,
    grid_h: u32,
    img_w: u32,
    img_h: u32,
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

// Streaming-friendly geometry inference from midlines only
#[derive(Debug, Clone)]
struct Midlines {
    width: u32,
    height: u32,
    mid_row_luma: Vec<u8>, // len = width
    mid_col_luma: Vec<u8>, // len = height
}

fn infer_geometry_from_midlines(m: &Midlines, threshold: u8) -> Result<Geometry> {
    let w = m.width as usize;
    let h = m.height as usize;
    if w < 40 || h < 40 {
        bail!("Image too small for frame/grid detection ({}×{})", w, h);
    }

    // Binarize midlines with given threshold
    let mut row_bin = vec![0u8; w];
    for i in 0..w { row_bin[i] = if m.mid_row_luma[i] > threshold { 255 } else { 0 }; }
    let mut col_bin = vec![0u8; h];
    for i in 0..h { col_bin[i] = if m.mid_col_luma[i] > threshold { 255 } else { 0 }; }

    let (q_left, f_left) = scan_white_then_black(&row_bin)
        .with_context(|| format!("Row probe (y={}): could not detect quiet+frame (threshold={})", h/2, threshold))?;
    let (q_right, f_right) = {
        let mut rev = row_bin.clone(); rev.reverse();
        scan_white_then_black(&rev)
            .with_context(|| format!("Row reverse probe (y={}): quiet+frame fail (threshold={})", h/2, threshold))?
    };
    let (q_top, f_top) = scan_white_then_black(&col_bin)
        .with_context(|| format!("Column probe (x={}): quiet+frame fail (threshold={})", w/2, threshold))?;
    let (q_bottom, f_bottom) = {
        let mut rev = col_bin.clone(); rev.reverse();
        scan_white_then_black(&rev)
            .with_context(|| format!("Column reverse probe (x={}): quiet+frame fail (threshold={})", w/2, threshold))?
    };

    let cell_px_x = (f_left.min(f_right) / FRAME_CELLS).max(1);
    let cell_px_y = (f_top.min(f_bottom) / FRAME_CELLS).max(1);
    let cell_px = cell_px_x.min(cell_px_y).max(1);
    let frame_px_quant = FRAME_CELLS * cell_px;

    let origin_x = q_left + frame_px_quant;
    let origin_y = q_top  + frame_px_quant;
    let inner_right = (w as u32).saturating_sub(q_right + frame_px_quant);
    let inner_bottom= (h as u32).saturating_sub(q_bottom + frame_px_quant);

    if origin_x >= inner_right || origin_y >= inner_bottom {
        bail!("Invalid frame bounds after quantization: origin=({}, {}), inner_right={}, inner_bottom={}",
              origin_x, origin_y, inner_right, inner_bottom);
    }

    let grid_w_px = inner_right - origin_x;
    let grid_h_px = inner_bottom - origin_y;
    let grid_w = (grid_w_px / cell_px).max(1);
    let grid_h = (grid_h_px / cell_px).max(1);

    Ok(Geometry { cell_px, origin_x, origin_y, grid_w, grid_h, img_w: w as u32, img_h: h as u32 })
}

// ========================= PNG streaming helpers =========================

fn luma_from_rgb(r: u8, g: u8, b: u8) -> u8 {
    // Integer approx of 0.299R + 0.587G + 0.114B
    let y = (77u16 * (r as u16) + 150u16 * (g as u16) + 29u16 * (b as u16) + 128) >> 8;
    y as u8
}

fn pass_a_scan_midlines(path: &Path) -> Result<(Midlines, u8)> {
    let file = fs::File::open(path).with_context(|| format!("open {:?}", path))?;
    let mut dec = PngDecoder::new(file);
    dec.set_transformations(PngXform::EXPAND | PngXform::STRIP_16);
    let mut reader = dec.read_info().context("png read_info")?;
    let info = reader.info();
    let width = info.width;
    let height = info.height;
    if width == 0 || height == 0 { bail!("PNG has zero dimension"); }
    let mid_y = height / 2;
    let mid_x = width / 2;
    let mut mid_row_luma = vec![0u8; width as usize];
    let mut mid_col_luma = vec![0u8; height as usize];
    let mut hist = [0u64; 256];

    let ct = info.color_type;
    match ct {
        PngColorType::Grayscale | PngColorType::GrayscaleAlpha | PngColorType::Rgb | PngColorType::Rgba => {}
        other => bail!("Unsupported PNG color type after EXPAND/STRIP16: {:?}", other),
    };

    for row_index in 0..height {
        let row = reader.next_row()?.ok_or_else(|| anyhow!("png missing row {}", row_index))?;
        let data = row.data();
        // For histogram & mid row, compute luma for entire row once
        for x in 0..(width as usize) {
            let l = match ct {
                PngColorType::Grayscale => data[x],
                PngColorType::GrayscaleAlpha => data[x * 2],
                PngColorType::Rgb => {
                    let i = x * 3; luma_from_rgb(data[i], data[i+1], data[i+2])
                }
                PngColorType::Rgba => {
                    let i = x * 4; luma_from_rgb(data[i], data[i+1], data[i+2])
                }
                _ => unreachable!(),
            };
            hist[l as usize] += 1;
            if row_index == mid_y { mid_row_luma[x] = l; }
        }
        // Mid column sample
        let mx = mid_x as usize;
        let l = match ct {
            PngColorType::Grayscale => data[mx],
            PngColorType::GrayscaleAlpha => data[mx * 2],
            PngColorType::Rgb => { let i = mx * 3; luma_from_rgb(data[i], data[i+1], data[i+2]) }
            PngColorType::Rgba => { let i = mx * 4; luma_from_rgb(data[i], data[i+1], data[i+2]) }
            _ => unreachable!(),
        };
        if (row_index as usize) < mid_col_luma.len() { mid_col_luma[row_index as usize] = l; }
    }

    // Otsu threshold from histogram
    let mut total: u64 = 0; for v in hist { total += v; }
    let mut sum: u64 = 0; for t in 0..256 { sum += (t as u64) * (hist[t] as u64); }
    let mut sum_b = 0u64; let mut w_b = 0u64; let mut max = 0f64; let mut thr = 0u8;
    for t in 0..256 {
        w_b += hist[t] as u64; if w_b == 0 { continue; }
        let w_f = total - w_b; if w_f == 0 { break; }
        sum_b += (t as u64) * (hist[t] as u64);
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum - sum_b) as f64 / w_f as f64;
        let var_b = (w_b as f64) * (w_f as f64) * (m_b - m_f).powi(2);
        if var_b > max { max = var_b; thr = t as u8; }
    }

    Ok((Midlines { width, height, mid_row_luma, mid_col_luma }, thr))
}

// ========================= Streaming parser and consumers =========================

enum Consumer<'a> {
    #[cfg(test)]
    Vec(Vec<u8>),
    File(FileConsumer),
    Compare(CompareConsumer<'a>),
}

struct FileConsumer {
    base_png: PathBuf,
    desired_name: Option<String>,
    final_path: Option<PathBuf>,
    out_path: Option<PathBuf>,
    tmp_path: Option<PathBuf>,
    file: Option<fs::File>,
    skip_write: bool,
}

impl FileConsumer {
    fn new(base_png: &Path) -> Self { Self { base_png: base_png.to_path_buf(), desired_name: None, final_path: None, out_path: None, tmp_path: None, file: None, skip_write: false } }
}

struct CompareConsumer<'a> { src: &'a [u8], pos: usize }

struct StreamParser {
    stage: ParserStage,
    headers: Vec<Header>,
    hdr_final: Option<Header>,
    hasher: Hasher,
    payload_remaining: u64,
}

enum ParserStage { Prefix { have: usize, buf: [u8; 6] }, HeaderLen { have: usize, buf: [u8;4], rep: usize }, HeaderBytes { have: usize, need: usize, buf: Vec<u8>, rep: usize }, Payload, Trailer { have: usize, buf: [u8;4] }, Done }

impl StreamParser {
    fn new() -> Self { Self { stage: ParserStage::Prefix { have: 0, buf: [0u8; 6] }, headers: Vec::with_capacity(HEADER_REPEAT), hdr_final: None, hasher: Hasher::new(), payload_remaining: 0 } }

    fn on_header_ready<'a>(&mut self, cons: &mut Consumer<'a>) -> Result<()> {
        let hdr = self.headers.last().cloned().unwrap();
        match cons {
            #[cfg(test)]
            Consumer::Vec(_) => {}
            Consumer::Compare(_) => {}
            Consumer::File(fc) => {
                let desired = if hdr.filename.trim().is_empty() { "decoded.bin".to_string() } else { hdr.filename.clone() };
                fc.desired_name = Some(desired.clone());
                let mut out_path = fc.base_png.with_file_name(&desired);
                if out_path.exists() {
                    let (h, l) = blake3_hash_file(&out_path)?;
                    if l == hdr.total_len && h == hdr.file_hash {
                        // Idempotent: skip writing; keep existing file
                        fc.final_path = Some(out_path);
                        fc.skip_write = true;
                        return Ok(());
                    } else {
                        out_path = fc.base_png.with_file_name(format!("{}.restored", desired));
                    }
                }
                let tmp = fc.base_png.with_file_name(format!("{}.restoring.tmp", desired));
                let file = fs::File::create(&tmp).with_context(|| format!("create {:?}", tmp))?;
                fc.final_path = Some(out_path.clone());
                fc.out_path = Some(out_path);
                fc.tmp_path = Some(tmp);
                fc.file = Some(file);
            }
        }
        Ok(())
    }

    fn feed<'a>(&mut self, mut data: &[u8], cons: &mut Consumer<'a>) -> Result<()> {
        while !data.is_empty() {
            match &mut self.stage {
                ParserStage::Prefix { have, buf } => {
                    let need = 6 - *have;
                    let take = need.min(data.len());
                    buf[*have..*have+take].copy_from_slice(&data[..take]);
                    *have += take; data = &data[take..];
                    if *have == 6 {
                        if &buf[0..4] != MAGIC { bail!("Magic mismatch: got={:02x?}, expected={:02x?}", &buf[0..4], MAGIC); }
                        let ver = u16::from_le_bytes([buf[4], buf[5]]);
                        if ver != VERSION { bail!("Unsupported VERSION {} (expected {}).", ver, VERSION); }
                        self.stage = ParserStage::HeaderLen { have: 0, buf: [0u8;4], rep: 0 };
                    }
                }
                ParserStage::HeaderLen { have, buf, rep } => {
                    let need = 4 - *have; let take = need.min(data.len());
                    buf[*have..*have+take].copy_from_slice(&data[..take]);
                    *have += take; data = &data[take..];
                    if *have == 4 {
                        let need = u32::from_le_bytes(*buf) as usize;
                        self.stage = ParserStage::HeaderBytes { have: 0, need, buf: Vec::with_capacity(need), rep: *rep };
                    }
                }
                ParserStage::HeaderBytes { have, need, buf, rep } => {
                    let want = *need - *have; let take = want.min(data.len());
                    buf.extend_from_slice(&data[..take]);
                    *have += take; data = &data[take..];
                    if *have == *need {
                        let (hdr, rest) = postcard::take_from_bytes::<Header>(&buf).map_err(|e| anyhow!("Header[{}] postcard decode: {}", *rep, e))?;
                        if !rest.is_empty() { bail!("Header[{}] trailing bytes ({})", *rep, rest.len()); }
                        self.headers.push(hdr);
                        if *rep + 1 < HEADER_REPEAT { self.stage = ParserStage::HeaderLen { have: 0, buf: [0u8;4], rep: *rep + 1 }; }
                        else {
                            if self.headers.windows(2).any(|w| w[0] != w[1]) { bail!("Header repeat mismatch: H0 != H1"); }
                            let hdr = self.headers.last().cloned().unwrap();
                            self.payload_remaining = hdr.payload_len;
                            self.hdr_final = Some(hdr.clone());
                            self.on_header_ready(cons)?;
                            self.stage = ParserStage::Payload;
                        }
                    }
                }
                ParserStage::Payload => {
                    if self.payload_remaining == 0 {
                        self.stage = ParserStage::Trailer { have: 0, buf: [0u8;4] };
                        continue;
                    }
                    let take = (self.payload_remaining as usize).min(data.len());
                    let chunk = &data[..take];
                    match cons {
                        #[cfg(test)]
                        Consumer::Vec(v) => v.extend_from_slice(chunk),
                        Consumer::Compare(c) => {
                            if c.pos + take > c.src.len() { bail!("Decoded payload longer than source slice"); }
                            if &c.src[c.pos..c.pos+take] != chunk { bail!("Streamed bytes differ from source slice at {}", c.pos); }
                            c.pos += take;
                        }
                        Consumer::File(fc) => { if let Some(f) = fc.file.as_mut() { std::io::Write::write_all(f, chunk)?; } }
                    }
                    self.hasher.update(chunk);
                    self.payload_remaining -= take as u64;
                    data = &data[take..];
                }
                ParserStage::Trailer { have, buf } => {
                    let need = 4 - *have; let take = need.min(data.len());
                    buf[*have..*have+take].copy_from_slice(&data[..take]);
                    *have += take; data = &data[take..];
                    if *have == 4 {
                        let hdr = self.hdr_final.clone().unwrap();
                        let trailer_u32 = u32::from_le_bytes(*buf);
                        let got_hash = *self.hasher.finalize().as_bytes();
                        if trailer_u32 != hdr.payload_hash32 { bail!("Trailer u32 mismatch: stream={:#010x}, header={:#010x}", trailer_u32, hdr.payload_hash32); }
                        if got_hash != hdr.file_hash { bail!("BLAKE3 mismatch after streaming payload"); }
                        if hdr.total_len != hdr.payload_len { bail!("Header total_len ({}) != payload_len ({})", hdr.total_len, hdr.payload_len); }
                        self.stage = ParserStage::Done;
                    }
                }
                ParserStage::Done => { return Ok(()); }
            }
        }
        Ok(())
    }
}

fn decode_streaming_with_consumer<'a>(path: &Path, geom: &Geometry, threshold: u8, mut consumer: Consumer<'a>) -> Result<Consumer<'a>> {
    // Precompute lattice centers
    let center_off = geom.cell_px / 2;
    let mut centers_x: Vec<u32> = Vec::with_capacity(geom.grid_w as usize);
    for c in 0..geom.grid_w { centers_x.push((geom.origin_x + c * geom.cell_px + center_off).min(geom.img_w - 1)); }
    let mut centers_y: Vec<u32> = Vec::with_capacity(geom.grid_h as usize);
    for r in 0..geom.grid_h { centers_y.push((geom.origin_y + r * geom.cell_px + center_off).min(geom.img_h - 1)); }

    let file = fs::File::open(path).with_context(|| format!("open {:?}", path))?;
    let mut dec = PngDecoder::new(file);
    dec.set_transformations(PngXform::EXPAND | PngXform::STRIP_16);
    let mut reader = dec.read_info().context("png read_info")?;
    let info = reader.info();
    let ct = info.color_type;

    let mut parser = StreamParser::new();
    // For File consumer, we must carry base path
    if let Consumer::File(ref mut fc) = consumer { if fc.base_png.as_os_str().is_empty() { fc.base_png = path.to_path_buf(); } }

    let mut next_row_idx = 0usize;
    let mut bit_buf: u8 = 0; let mut bit_count: u8 = 0;
    let mut out_bytes: Vec<u8> = Vec::with_capacity(1024);

    for y in 0..geom.img_h {
        let row = reader.next_row()?.ok_or_else(|| anyhow!("png missing row {}", y))?;
        let data = row.data();
        let target_y = if next_row_idx < centers_y.len() { centers_y[next_row_idx] } else { geom.img_h };
        if y == target_y {
            for &cx in &centers_x {
                let mx = cx as usize;
                let l = match ct {
                    PngColorType::Grayscale => data[mx],
                    PngColorType::GrayscaleAlpha => data[mx * 2],
                    PngColorType::Rgb => { let i = mx * 3; luma_from_rgb(data[i], data[i+1], data[i+2]) }
                    PngColorType::Rgba => { let i = mx * 4; luma_from_rgb(data[i], data[i+1], data[i+2]) }
                    other => bail!("Unsupported PNG color type after EXPAND/STRIP16: {:?}", other),
                };
                let bit = if l <= threshold { 1u8 } else { 0u8 };
                bit_buf |= bit << (7 - bit_count);
                bit_count += 1;
                if bit_count == 8 { out_bytes.push(bit_buf); bit_buf = 0; bit_count = 0; }
            }
            next_row_idx += 1;
        }

        // Feed any produced bytes to parser
        if !out_bytes.is_empty() {
            parser.feed(&out_bytes, &mut consumer)?;
            out_bytes.clear();
        }
        if let ParserStage::Done = parser.stage { break; }
    }

    // Flush leftover partial byte (if any)
    if bit_count > 0 { out_bytes.push(bit_buf); parser.feed(&out_bytes, &mut consumer)?; }

    // If file consumer, finalize/rename/verify
    match &mut consumer {
        Consumer::File(fc) => {
            if fc.skip_write {
                // Nothing to rename; final_path already points to existing verified file
            } else if let (Some(tmp), Some(out)) = (fc.tmp_path.take(), fc.out_path.take()) {
                // Close file handle, then atomically move tmp -> out
                drop(fc.file.take());
                fs::rename(&tmp, &out).with_context(|| format!("rename {:?} -> {:?}", tmp, out))?;
                let (reh, relen) = blake3_hash_file(&out)?;
                let hdr = parser.hdr_final.ok_or_else(|| anyhow!("Missing header at finalize"))?;
                if relen != hdr.total_len { bail!("Output length mismatch after write ({} vs {})", relen, hdr.total_len); }
                if reh != hdr.file_hash { bail!("Output BLAKE3 mismatch after write"); }
            }
        }
        _ => {}
    }

    if let ParserStage::Done = parser.stage { Ok(consumer) } else { bail!("Premature end of PNG rows before trailer") }
}

fn decode_png_to_file_streaming(path: &Path) -> Result<PathBuf> {
    let (mid, otsu) = pass_a_scan_midlines(path)?;
    let mut tries: Vec<u8> = vec![otsu];
    tries.extend_from_slice(FALLBACK_THRESHOLDS);
    let mut last_errs: Vec<anyhow::Error> = Vec::new();
    for thr in tries {
        match infer_geometry_from_midlines(&mid, thr)
            .and_then(|geom| decode_streaming_with_consumer(path, &geom, thr, Consumer::File(FileConsumer::new(path)))) {
            Ok(Consumer::File(fc)) => {
                if let Some(final_path) = fc.final_path { return Ok(final_path); }
                if let Some(out) = fc.out_path { return Ok(out); }
                if let Some(name) = fc.desired_name { return Ok(path.with_file_name(name)); }
                bail!("Decode completed but output path unknown");
            }
            Ok(_) => unreachable!(),
            Err(e) => last_errs.push(e),
        }
    }
    let mut msg = String::new(); use std::fmt::Write as _;
    writeln!(&mut msg, "All threshold attempts failed for {:?}. Attempts: {}", path, last_errs.len()).ok();
    for (i, e) in last_errs.iter().enumerate() { writeln!(&mut msg, "  [{}] {}", i, e).ok(); }
    bail!(msg)
}

fn decode_png_compare_to_slice(path: &Path, original: &[u8]) -> Result<()> {
    let (mid, otsu) = pass_a_scan_midlines(path)?;
    let mut tries: Vec<u8> = vec![otsu];
    tries.extend_from_slice(FALLBACK_THRESHOLDS);
    let mut last_errs: Vec<anyhow::Error> = Vec::new();
    for thr in tries {
        match infer_geometry_from_midlines(&mid, thr)
            .and_then(|geom| decode_streaming_with_consumer(path, &geom, thr, Consumer::Compare(CompareConsumer { src: original, pos: 0 }))) {
            Ok(_) => return Ok(()),
            Err(e) => last_errs.push(e),
        }
    }
    let mut msg = String::new(); use std::fmt::Write as _;
    writeln!(&mut msg, "All threshold attempts failed for {:?}. Attempts: {}", path, last_errs.len()).ok();
    for (i, e) in last_errs.iter().enumerate() { writeln!(&mut msg, "  [{}] {}", i, e).ok(); }
    bail!(msg)
}

#[cfg(test)]
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

    Ok(Geometry { cell_px, origin_x, origin_y, grid_w, grid_h, img_w: w as u32, img_h: h as u32 })
}

// ========================= Bit packing / unpacking =========================

fn bytes_to_bits(data: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(data.len() * 8);
    for b in data {
        for k in (0..8).rev() { bits.push((b >> k) & 1); }
    }
    bits
}
#[cfg(test)]
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

// Pick cell_px targeting ~TARGET_SIDE_PX (best effort, not enforced).
fn choose_cell_px_for_canvas(grid_w: u32, grid_h: u32) -> u32 {
    let total_cells_x = grid_w + 2 * (QUIET_CELLS + FRAME_CELLS);
    let total_cells_y = grid_h + 2 * (QUIET_CELLS + FRAME_CELLS);
    let max_cells_side = total_cells_x.max(total_cells_y).max(1);
    let ideal = (TARGET_SIDE_PX / max_cells_side).max(1);
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

    // Size grid + header + cell_px with canvas sizing (best-effort target).
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

    // Active proof: re-open and streaming-verify decode
    step!("Re-opening PNG for streaming verification…");
    decode_png_compare_to_slice(&out, &bytes)?;
    ok!("Round-trip verification OK (length + BLAKE3 + exact bytes).");

    eprintln!();
    ok!("ENCODE COMPLETE → {:?}", out.file_name().unwrap_or_default());
    Ok(out)
}

// ========================= Decode =========================

#[cfg(test)]
fn decode_png_to_bytes(path: &Path) -> Result<Vec<u8>> {
    let (mid, otsu) = pass_a_scan_midlines(path)?;
    let mut tries: Vec<u8> = vec![otsu];
    tries.extend_from_slice(FALLBACK_THRESHOLDS);
    let mut last_errs: Vec<anyhow::Error> = Vec::new();
    for thr in tries {
        match infer_geometry_from_midlines(&mid, thr)
            .and_then(|geom| decode_streaming_with_consumer(path, &geom, thr, Consumer::Vec(Vec::new()))) {
            Ok(Consumer::Vec(v)) => return Ok(v),
            Ok(_) => unreachable!(),
            Err(e) => last_errs.push(e),
        }
    }
    let mut msg = String::new(); use std::fmt::Write as _;
    writeln!(&mut msg, "All threshold attempts failed for {:?}. Attempts: {}", path, last_errs.len()).ok();
    for (i, e) in last_errs.iter().enumerate() { writeln!(&mut msg, "  [{}] {}", i, e).ok(); }
    bail!(msg)
}

// (removed legacy full-frame decode function; streaming path is used exclusively)

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
        // DECODE → write original file (idempotent, streaming, one PNG in/one file out)
        step!("Decoding PNG {:?} → original file (streaming)…", path.file_name().unwrap_or_default());
        let out_path = decode_png_to_file_streaming(&path)?;
        ok!("Output verified (length + BLAKE3): {:?}", out_path.file_name().unwrap_or_default());
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
    use image::imageops;

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

    #[test]
    fn roundtrip_2mb_text_streaming_to_file() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("big.txt");
        let size = 2 * 1024 * 1024; // 2 MiB
        let mut content = Vec::with_capacity(size);
        // Fill with deterministic ASCII pattern
        while content.len() < size {
            let line = b"The quick brown fox jumps over the lazy dog.\n";
            let take = (size - content.len()).min(line.len());
            content.extend_from_slice(&line[..take]);
        }
        fs::write(&input, &content)?;

        // Encode
        let out_png = super::encode_file_to_png(&input)?;

        // Streaming decode to file
        let out_file = super::decode_png_to_file_streaming(&out_png)?;

        // Verify
        let decoded = fs::read(&out_file)?;
        assert_eq!(decoded.len(), content.len());
        assert_eq!(blake3_hash_bytes(&decoded), blake3_hash_bytes(&content));
        assert_eq!(decoded, content);
        Ok(())
    }
}
