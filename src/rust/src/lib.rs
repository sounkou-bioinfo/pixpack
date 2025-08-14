// Minimal PixPack bindings for R via savvy
// Copied core encode/decode logic (trimmed) from pixpack CLI.
// Logging macros converted to no-ops to stay silent in R sessions.

use anyhow::{anyhow, bail, Context, Result};
use blake3::Hasher;
use image::{DynamicImage, GrayImage, ImageBuffer, Rgba, RgbaImage, RgbImage};
use serde::{Deserialize, Serialize};
use std::{ffi::OsStr, fs, io::Read, path::{Path, PathBuf}, time::{SystemTime, UNIX_EPOCH}};
use savvy::savvy;
use savvy::{NotAvailableValue, OwnedStringSexp, StringSexp};
use png::{Decoder as PngDecoder, ColorType as PngColorType};

// ---------------- Configuration ----------------
const MAGIC: &[u8; 4] = b"PXPV"; // PixPack Visual v1
const VERSION: u16 = 1;
const QUIET_CELLS: u32 = 3;
const FRAME_CELLS: u32 = 2;
const MIN_CELL_PX: u32 = 6;
const MAX_CELL_PX: u32 = 20;
const MAX_SIDE_PX: u32 = 5000;
const HEADER_REPEAT: usize = 2;
const FALLBACK_THRESHOLDS: &[u8] = &[200, 180, 160, 140, 120, 100];

macro_rules! step { ($($arg:tt)*) => {}; }
macro_rules! ok { ($($arg:tt)*) => {}; }
macro_rules! warn { ($($arg:tt)*) => {}; }

// ---------------- Types & integrity ----------------
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
    grid_w: u32,
    grid_h: u32,
    quiet_cells: u32,
    frame_cells: u32,
    payload_len: u64,
    payload_hash32: u32,
}

fn blake3_hash_bytes(bytes: &[u8]) -> [u8; 32] { let mut h = Hasher::new(); h.update(bytes); *h.finalize().as_bytes() }
fn blake3_hash_file(path: &Path) -> Result<([u8; 32], u64)> {
    let mut f = fs::File::open(path)?; let mut h = Hasher::new(); let mut buf = vec![0u8; 1<<20]; let mut total=0u64;
    loop { let n = f.read(&mut buf)?; if n==0 { break; } h.update(&buf[..n]); total += n as u64; }
    Ok((*h.finalize().as_bytes(), total))
}
fn verify_bytes(expected_len: u64, expected_hash: [u8; 32], data: UnverifiedBytes) -> Result<VerifiedBytes> {
    if data.0.len() as u64 != expected_len { bail!("Length mismatch: expected {}, got {}", expected_len, data.0.len()); }
    let got = blake3_hash_bytes(&data.0); if got != expected_hash { bail!("BLAKE3 mismatch"); }
    Ok(VerifiedBytes(data.0))
}
fn hash32_first(bytes: &[u8]) -> u32 { let h = blake3_hash_bytes(bytes); u32::from_le_bytes([h[0],h[1],h[2],h[3]]) }
fn derive_dataset_id(bytes: &[u8], filename: &str) -> [u8; 16] {
    let mut h = Hasher::new(); h.update(bytes); h.update(filename.as_bytes()); h.update(&(bytes.len() as u64).to_le_bytes());
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos(); h.update(&ts.to_le_bytes());
    let digest = h.finalize(); let mut id=[0u8;16]; id.copy_from_slice(&digest.as_bytes()[..16]); id
}

// ---------------- Otsu & binarization ----------------
fn otsu_threshold(gray: &GrayImage) -> u8 {
    let mut hist=[0u64;256]; for p in gray.pixels(){ hist[p[0] as usize]+=1; } let total: u64 = hist.iter().sum();
    let mut sum=0u64; for t in 0..256 { sum += (t as u64)*hist[t]; }
    let mut sum_b=0u64; let mut w_b=0u64; let mut max=0.0f64; let mut thresh=0u8;
    for t in 0..256 { w_b+=hist[t]; if w_b==0 { continue; } let w_f= total - w_b; if w_f==0 { break; }
        sum_b += (t as u64)*hist[t]; let m_b = sum_b as f64 / w_b as f64; let m_f = (sum - sum_b) as f64 / w_f as f64;
        let var_between = (w_b as f64)*(w_f as f64)*(m_b - m_f).powi(2); if var_between > max { max=var_between; thresh=t as u8; } }
    thresh
}
fn binarize(gray: &GrayImage, t: u8) -> GrayImage {
    let mut out = gray.clone(); for p in out.pixels_mut(){ p.0[0] = if p.0[0] > t {255} else {0}; } out
}

// ---------------- Geometry inference ----------------
#[derive(Debug, Clone)]
struct Geometry { threshold: u8, cell_px: u32, origin_x: u32, origin_y: u32, grid_w: u32, grid_h: u32, img_w: u32, img_h: u32 }

fn scan_white_then_black(line: &[u8]) -> Result<(u32,u32)> { let mut i=0usize; while i<line.len() && line[i]==255 { i+=1; } if i==line.len(){ bail!("No black after white"); } let q=i as u32; let mut j=i; while j<line.len() && line[j]==0 { j+=1; } let f=(j-i) as u32; if f==0 { bail!("Frame thickness 0"); } Ok((q,f)) }

fn try_infer_geometry(bin:&GrayImage, threshold:u8) -> Result<Geometry> {
    let w=bin.width() as usize; let h=bin.height() as usize; if w<40 || h<40 { bail!("Image too small"); }
    let mid_y = h/2; let row=&bin.as_raw()[mid_y*w .. (mid_y+1)*w]; let (q_left,f_left)=scan_white_then_black(row)?; let (q_right,f_right)={ let mut rev=row.to_vec(); rev.reverse(); scan_white_then_black(&rev)? };
    let mid_x = w/2; let mut col=Vec::with_capacity(h); for y in 0..h { col.push(bin.as_raw()[y*w+mid_x]); } let (q_top,f_top)=scan_white_then_black(&col)?; let (q_bottom,f_bottom)={ let mut rev=col.clone(); rev.reverse(); scan_white_then_black(&rev)? };
    let cell_px_x=(f_left.min(f_right)/FRAME_CELLS).max(1); let cell_px_y=(f_top.min(f_bottom)/FRAME_CELLS).max(1); let cell_px=cell_px_x.min(cell_px_y).max(1); let frame_px_quant = FRAME_CELLS*cell_px;
    let origin_x = q_left + frame_px_quant; let origin_y = q_top + frame_px_quant; let inner_right = (w as u32).saturating_sub(q_right + frame_px_quant); let inner_bottom = (h as u32).saturating_sub(q_bottom + frame_px_quant);
    if origin_x >= inner_right || origin_y >= inner_bottom { bail!("Invalid frame bounds"); }
    let grid_w_px = inner_right - origin_x; let grid_h_px = inner_bottom - origin_y; let grid_w = (grid_w_px / cell_px).max(1); let grid_h = (grid_h_px / cell_px).max(1);
    Ok(Geometry { threshold, cell_px, origin_x, origin_y, grid_w, grid_h, img_w: w as u32, img_h: h as u32 })
}

// ---------------- Bit packing ----------------
fn bytes_to_bits(data:&[u8])->Vec<u8>{ let mut bits=Vec::with_capacity(data.len()*8); for b in data { for k in (0..8).rev(){ bits.push((b>>k)&1); } } bits }
fn bits_to_bytes(bits:&[u8])->Vec<u8>{ let mut out=vec![0u8; (bits.len()+7)/8]; for (i,bit) in bits.iter().enumerate(){ if *bit!=0 { let byte=i/8; let off=7-(i%8); out[byte]|=1<<off; } } out }

// ---------------- Grid sizing ----------------
fn compute_overhead(header_len:usize)->usize { 4+2 + HEADER_REPEAT*(4+header_len) + 4 }
fn minimal_grid(bits_needed:usize)->(u32,u32){ let side=(bits_needed as f64).sqrt().ceil() as u32; let w=side.max(1); let total=bits_needed as u128; let h=((total + (w as u128)-1)/(w as u128)) as u32; (w,h.max(1)) }
fn choose_cell_px_for_canvas(grid_w:u32, grid_h:u32)->u32 { let total_cells_x=grid_w + 2*(QUIET_CELLS+FRAME_CELLS); let total_cells_y=grid_h + 2*(QUIET_CELLS+FRAME_CELLS); let max_side= total_cells_x.max(total_cells_y).max(1); let ideal=(MAX_SIDE_PX / max_side).max(1); ideal.clamp(MIN_CELL_PX, MAX_CELL_PX).max(1) }

// ---------------- Encode ----------------
fn choose_grid_and_header(filename:&str, bytes:&[u8], dataset_id:[u8;16]) -> Result<(Header, Vec<u8>, usize, u32)> {
    let mut grid_w=1u32; let mut grid_h=1u32;
    for _ in 0..6 { let header_tmp = Header { version: VERSION, filename: filename.to_string(), total_len: bytes.len() as u64, file_hash: blake3_hash_bytes(bytes), dataset_id, grid_w, grid_h, quiet_cells: QUIET_CELLS, frame_cells: FRAME_CELLS, payload_len: bytes.len() as u64, payload_hash32: hash32_first(bytes) };
        let hdr_bytes = postcard::to_allocvec(&header_tmp)?; let overhead = compute_overhead(hdr_bytes.len()); let needed_bits = (overhead + bytes.len()) * 8;
        let (gw,gh)=minimal_grid(needed_bits); let _cell_px=choose_cell_px_for_canvas(gw,gh); let header_final = Header { grid_w: gw, grid_h: gh, ..header_tmp }; let hdr_final_bytes = postcard::to_allocvec(&header_final)?; let overhead2 = compute_overhead(hdr_final_bytes.len()); let needed2=(overhead2 + bytes.len())*8; if (gw as usize)*(gh as usize) >= needed2 { let cell_px=choose_cell_px_for_canvas(gw,gh); return Ok((header_final, hdr_final_bytes, overhead2, cell_px)); } grid_w=gw; grid_h=gh; }
    bail!("Unable to size grid/header")
}

fn render_stream_to_image(stream:&[u8], grid_w:u32, grid_h:u32, cell_px:u32) -> Result<RgbaImage> {
    let total_cells_x = grid_w + 2*(QUIET_CELLS+FRAME_CELLS); let total_cells_y = grid_h + 2*(QUIET_CELLS+FRAME_CELLS); let width_px = total_cells_x * cell_px; let height_px = total_cells_y * cell_px;
    let mut img: RgbaImage = ImageBuffer::from_pixel(width_px, height_px, Rgba([255,255,255,255]));
    let left = QUIET_CELLS*cell_px; let top=QUIET_CELLS*cell_px; let right = width_px - QUIET_CELLS*cell_px; let bottom = height_px - QUIET_CELLS*cell_px;
    for y in top..(top + FRAME_CELLS*cell_px) { for x in left..right { img.put_pixel(x,y,Rgba([0,0,0,255])); } }
    for y in (bottom - FRAME_CELLS*cell_px)..bottom { for x in left..right { img.put_pixel(x,y,Rgba([0,0,0,255])); } }
    for y in (top + FRAME_CELLS*cell_px)..(bottom - FRAME_CELLS*cell_px) {
        for x in left..(left + FRAME_CELLS*cell_px) { img.put_pixel(x,y,Rgba([0,0,0,255])); }
        for x in (right - FRAME_CELLS*cell_px)..right { img.put_pixel(x,y,Rgba([0,0,0,255])); }
    }
    let origin_x = left + FRAME_CELLS*cell_px; let origin_y = top + FRAME_CELLS*cell_px; let bits=bytes_to_bits(stream); let mut bit_index=0usize;
    for row in 0..grid_h { for col in 0..grid_w { if bit_index >= bits.len() { break; } if bits[bit_index]==1 { let x0=origin_x + col*cell_px; let y0=origin_y + row*cell_px; for y in y0..(y0+cell_px) { for x in x0..(x0+cell_px) { img.put_pixel(x,y,Rgba([0,0,0,255])); } } } bit_index += 1; } }
    Ok(img)
}

fn encode_file_to_png(input:&Path) -> Result<PathBuf> {
    let bytes = fs::read(input).with_context(|| format!("read {:?}", input))?; let (file_hash_stream, file_len_stream)=blake3_hash_file(input)?; if bytes.len() as u64 != file_len_stream { bail!("File changed (size)"); }
    let file_hash = blake3_hash_bytes(&bytes); if file_hash != file_hash_stream { bail!("File changed (hash)"); }
    let filename = input.file_name().and_then(OsStr::to_str).unwrap_or("unknown").to_string(); let dataset_id = derive_dataset_id(&bytes, &filename);
    let (header, hdr_bytes, overhead, cell_px)=choose_grid_and_header(&filename, &bytes, dataset_id)?; let cap_bits=(header.grid_w as usize)*(header.grid_h as usize); let needed_bits=(overhead + bytes.len())*8; assert!(cap_bits >= needed_bits);
    let mut stream=Vec::with_capacity(overhead + bytes.len()); stream.extend_from_slice(MAGIC); stream.extend_from_slice(&VERSION.to_le_bytes()); let hdr_len_le=(hdr_bytes.len() as u32).to_le_bytes(); for _ in 0..HEADER_REPEAT { stream.extend_from_slice(&hdr_len_le); stream.extend_from_slice(&hdr_bytes); } stream.extend_from_slice(&bytes); stream.extend_from_slice(&header.payload_hash32.to_le_bytes());
    let img = render_stream_to_image(&stream, header.grid_w, header.grid_h, cell_px)?; let out = if let Some(ext)= input.extension().and_then(OsStr::to_str) { input.with_extension(format!("{ext}.png")) } else { input.with_extension("png") };
    DynamicImage::ImageRgba8(img).save(&out).context("write png")?; // Roundtrip verification
    let decoded = decode_png_to_bytes(&out)?; let verified = verify_bytes(header.total_len, header.file_hash, UnverifiedBytes(decoded))?; if verified.as_slice() != bytes.as_slice() { bail!("Post-write mismatch"); }
    Ok(out)
}

// ---------------- Decode ----------------
fn open_to_gray(path: &Path) -> Result<GrayImage> {
    match image::open(path) {
        Ok(d) => Ok(d.to_luma8()),
        Err(_) => {
            let file = fs::File::open(path)?;
            let decoder = PngDecoder::new(file);
            let mut reader = decoder.read_info().context("png read_info failed")?;
            let mut buf = vec![0u8; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf).context("png next_frame failed")?;
            let w = info.width;
            let h = info.height;
            let size = info.buffer_size();
            let data = &buf[..size];
            match info.color_type {
                PngColorType::Grayscale => {
                    GrayImage::from_vec(w, h, data.to_vec()).ok_or_else(|| anyhow!("invalid grayscale buffer"))
                }
                PngColorType::Rgb => {
                    let img = RgbImage::from_raw(w, h, data.to_vec()).ok_or_else(|| anyhow!("invalid rgb buffer"))?;
                    Ok(DynamicImage::ImageRgb8(img).to_luma8())
                }
                PngColorType::Rgba => {
                    let img = RgbaImage::from_raw(w, h, data.to_vec()).ok_or_else(|| anyhow!("invalid rgba buffer"))?;
                    Ok(DynamicImage::ImageRgba8(img).to_luma8())
                }
                _ => bail!("unsupported PNG color type"),
            }
        }
    }
}

fn decode_png_to_bytes(path:&Path) -> Result<Vec<u8>> {
    let gray0 = open_to_gray(path).with_context(|| format!("open {:?}", path))?;
    let otsu = otsu_threshold(&gray0); let mut tries: Vec<u8> = vec![otsu]; tries.extend_from_slice(FALLBACK_THRESHOLDS); let mut last_errs: Vec<anyhow::Error>=Vec::new();
    for thr in tries { let bin = binarize(&gray0, thr); match decode_from_binary_image_with_threshold(&bin, thr) { Ok(bytes)=> return Ok(bytes), Err(e)=> { last_errs.push(e); continue; } } }
    let mut msg=String::new(); use std::fmt::Write; writeln!(&mut msg, "All threshold attempts failed for {:?}. Attempts: {}", path, last_errs.len()).ok(); for (i,e) in last_errs.iter().enumerate(){ writeln!(&mut msg, "  [{}] {}", i, e).ok(); } bail!(msg)
}

fn decode_from_binary_image_with_threshold(bin:&GrayImage, threshold:u8) -> Result<Vec<u8>> {
    let geom = try_infer_geometry(bin, threshold)?; let center_off = geom.cell_px / 2; let raw=bin.as_raw(); let w=bin.width() as usize; let mut bits=Vec::with_capacity((geom.grid_w*geom.grid_h) as usize);
    for r in 0..geom.grid_h { for c in 0..geom.grid_w { let cx=(geom.origin_x + c*geom.cell_px + center_off).min(bin.width()-1) as usize; let cy=(geom.origin_y + r*geom.cell_px + center_off).min(bin.height()-1) as usize; let v=raw[cy*w + cx]; bits.push(if v < 128 {1} else {0}); } }
    let stream = bits_to_bytes(&bits); if stream.len() < 6 { bail!("Stream too short"); } if &stream[0..4] != MAGIC { bail!("Magic mismatch"); }
    let ver = u16::from_le_bytes([stream[4],stream[5]]); if ver != VERSION { bail!("Unsupported version"); }
    let mut cursor=6usize; let mut headers=Vec::with_capacity(HEADER_REPEAT);
    for i in 0..HEADER_REPEAT { if cursor + 4 > stream.len() { bail!("Header len missing {}", i); } let len = u32::from_le_bytes([stream[cursor],stream[cursor+1],stream[cursor+2],stream[cursor+3]]) as usize; cursor += 4; if cursor + len > stream.len() { bail!("Header truncated {}", i); } let (h, rest)=postcard::take_from_bytes::<Header>(&stream[cursor..cursor+len]).map_err(|e| anyhow!("Header decode {}: {e}", i))?; if !rest.is_empty() { bail!("Header trailing"); } headers.push(h); cursor += len; }
    if headers.windows(2).any(|w| w[0] != w[1]) { bail!("Header repeat mismatch"); } let hdr = headers.pop().unwrap();
    let payload_start = cursor; let payload_len = hdr.payload_len as usize; if payload_start + payload_len + 4 > stream.len() { bail!("Payload OOB"); }
    let payload = &stream[payload_start .. payload_start + payload_len]; let trailer = &stream[payload_start + payload_len .. payload_start + payload_len + 4]; let trailer_u32 = u32::from_le_bytes([trailer[0],trailer[1],trailer[2],trailer[3]]); let _got32 = hash32_first(payload);
    if trailer_u32 != hdr.payload_hash32 { bail!("Trailer mismatch"); } if blake3_hash_bytes(payload) != hdr.file_hash { bail!("Payload hash mismatch"); } if hdr.total_len != hdr.payload_len { bail!("Length mismatch header"); }
    Ok(payload.to_vec())
}

fn is_png(path:&Path) -> bool { match fs::File::open(path) { Ok(mut f) => { let mut sig=[0u8;8]; std::io::Read::read_exact(&mut f, &mut sig).is_ok() && &sig == b"\x89PNG\r\n\x1a\n" }, Err(_)=> false } }

// ---------------- Savvy-exported functions ----------------

/// Encode a file to PixPack PNG or decode a PixPack PNG back to original file.
///
/// @param path Input file path (character scalar).
/// @return Output artifact path (character scalar).
/// @export
#[savvy]
fn fileConversion(path: StringSexp) -> savvy::Result<savvy::Sexp> {
    let in_path = path.iter().next().ok_or_else(|| savvy::Error::new("path missing"))?;
    if in_path.is_na() { return Err(savvy::Error::new("path is NA")); }
    let p = PathBuf::from(in_path.to_string());
    if !p.exists() { return Err(savvy::Error::new("file does not exist")); }
    let out_path = if is_png(&p) { // decode and write original file next to PNG
        let bytes = decode_png_to_bytes(&p).map_err(|e| savvy::Error::new(e.to_string()))?;
        // Attempt to recover original filename from headers (quick parse reused)
        // For now, write as <stem>.decoded unless original file exists identical.
        let mut candidate = p.with_extension(""); // remove .png layer
        if candidate.extension().is_none() { candidate = p.with_extension("decoded.bin"); }
        if candidate.exists() { // if identical content skip
            if let Ok((h,l)) = blake3_hash_file(&candidate) { if l as usize == bytes.len() && h == blake3_hash_bytes(&bytes) { return string_out(candidate); } }
            candidate = p.with_file_name(format!("{}-restored", candidate.file_name().and_then(|s| s.to_str()).unwrap_or("restored")));
        }
        fs::write(&candidate, &bytes).map_err(|e| savvy::Error::new(e.to_string()))?;
        candidate
    } else { encode_file_to_png(&p).map_err(|e| savvy::Error::new(e.to_string()))? };
    string_out(out_path)
}

/// Encode a string into a PixPack PNG at given path or decode an existing PixPack PNG to string.
/// If `png_path` already exists and is a PixPack PNG, it is decoded; otherwise `input` is encoded to that PNG path.
///
/// @param input Character scalar (content when encoding; ignored on decode).
/// @param png_path Target PNG path (must be provided).
/// @return On encode: PNG path; On decode: decoded UTF-8 string (lossy).
/// @export
#[savvy]
fn StringConversion(input: StringSexp, png_path: StringSexp) -> savvy::Result<savvy::Sexp> {
    let content = input.iter().next().and_then(|s| if s.is_na(){None}else{Some(s.to_string())}).unwrap_or_default();
    let png_path_str = png_path.iter().next().ok_or_else(|| savvy::Error::new("png_path missing"))?;
    if png_path_str.is_na() { return Err(savvy::Error::new("png_path is NA")); }
    let p = PathBuf::from(png_path_str.to_string());
    if p.exists() && is_png(&p) { // decode → string
        let bytes = decode_png_to_bytes(&p).map_err(|e| savvy::Error::new(e.to_string()))?;
        let s = match String::from_utf8(bytes) { Ok(v)=>v, Err(e)=>String::from_utf8_lossy(e.as_bytes()).to_string() };
        let mut out = OwnedStringSexp::new(1)?; out.set_elt(0, &s)?; return Ok(out.into());
    }
    // encode string
    let tmp = std::env::temp_dir().join("_pixpack_string_input.tmp"); fs::write(&tmp, content.as_bytes()).map_err(|e| savvy::Error::new(e.to_string()))?;
    let png_out = encode_file_to_png(&tmp).map_err(|e| savvy::Error::new(e.to_string()))?;
    // rename/move to requested path if different
    if png_out != p { fs::rename(&png_out, &p).map_err(|e| savvy::Error::new(e.to_string()))?; }
    string_out(p)
}

fn string_out(path: PathBuf) -> savvy::Result<savvy::Sexp> { let mut out = OwnedStringSexp::new(1)?; out.set_elt(0, path.to_string_lossy().as_ref())?; Ok(out.into()) }
