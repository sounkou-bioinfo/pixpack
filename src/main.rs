// main.rs
// pixpack — encode arbitrary file -> viewable PNG (custom ancillary chunk), or
// decode such a PNG -> original file, with *active proofs* of integrity.
// One arg only: if path is PNG (by signature), decode; otherwise encode.
//
// Integrity proofs:
// - Header (magic, version, original filename, original length, blake3) serialized via postcard
// - On encode: after writing PNG, re-open, parse chunk, verify hash/length, exact bytes match
// - On decode: verify hash/length, write output, re-hash output
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
use image::{DynamicImage, GenericImage, Rgba};
use png_achunk as achunk;
use serde::{Deserialize, Serialize};

// -------------------- constants & types --------------------

const PNG_SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
const CHUNK_NAME: &str = "ruSt"; // ancillary/private/reserved/safe-to-copy
const MAGIC: &[u8; 8] = b"PNGPACK!";
const VERSION: u16 = 1;

/// Bytes that haven't been verified yet.
#[derive(Debug, Clone)]
struct UnverifiedBytes(Vec<u8>);

/// Bytes verified for exact length and BLAKE3 hash.
#[derive(Debug, Clone)]
struct VerifiedBytes(Vec<u8>);
impl VerifiedBytes {
    fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

/// Header stored before data inside the ancillary chunk.
#[derive(Debug, Serialize, Deserialize)]
struct PayloadHeader {
    magic: [u8; 8],
    version: u16,
    orig_filename: String,
    orig_len: u64,
    blake3: [u8; 32],
}
impl PayloadHeader {
    fn new(orig_filename: String, orig_len: u64, blake3: [u8; 32]) -> Self {
        let mut magic = [0u8; 8];
        magic.copy_from_slice(MAGIC);
        Self {
            magic,
            version: VERSION,
            orig_filename,
            orig_len,
            blake3,
        }
    }
    fn sanity_check(&self) -> Result<()> {
        if &self.magic != MAGIC {
            bail!("header magic mismatch");
        }
        if self.version != VERSION {
            bail!("unsupported header version: {} (expected {})", self.version, VERSION);
        }
        Ok(())
    }
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

// -------------------- helpers --------------------

fn is_png_file(path: &Path) -> Result<bool> {
    let mut f = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };
    let mut sig = [0u8; 8];
    if f.read_exact(&mut sig).is_ok() {
        return Ok(sig == PNG_SIG);
    }
    Ok(false)
}

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

fn split_header_and_data(chunk_bytes: &[u8]) -> Result<(PayloadHeader, &[u8])> {
    // postcard::take_from_bytes gives (T, remainder)
    let (hdr, rest) = postcard::take_from_bytes::<PayloadHeader>(chunk_bytes)
        .map_err(|e| anyhow!("deserializing header failed: {e}"))?;
    Ok((hdr, rest))
}

// -------------------- core encode/decode --------------------

fn encode_file_to_png(input: &Path) -> Result<PathBuf> {
    step!("Encoding file {:?} → PNG with embedded data…", input.file_name().unwrap_or_default());

    step!("Reading & hashing input…");
    let bytes = fs::read(input).with_context(|| format!("read {:?}", input))?;
    let (file_hash, file_len_stream) = blake3_hash_file(input)?;
    if bytes.len() as u64 != file_len_stream {
        bail!("file size changed during read ({} vs {})", bytes.len(), file_len_stream);
    }
    ok!("Input: {} bytes, BLAKE3={}", bytes.len(), hex32(file_hash));

    let fname = input
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("unknown")
        .to_string();

    let header = PayloadHeader::new(fname, bytes.len() as u64, file_hash);

    step!("Serializing header…");
    let header_bytes: Vec<u8> =
        postcard::to_allocvec(&header).context("postcard serialize header (enable `alloc` feature)")?;
    ok!("Header size: {} bytes", header_bytes.len());

    step!("Building ancillary chunk…");
    let ctype = achunk::ChunkType::from_ascii(&CHUNK_NAME).context("ChunkType::from_ascii")?;
    let mut payload = Vec::with_capacity(header_bytes.len() + bytes.len());
    payload.extend_from_slice(&header_bytes);
    payload.extend_from_slice(&bytes);
    let chunk = achunk::Chunk::new(ctype, payload).context("create chunk")?;
    ok!("Chunk ready ({} bytes payload).", chunk.data.len());

    // Create a tiny viewable image (8×8 RGBA) and paint a gradient
    step!("Creating 8×8 RGBA canvas…");
    let mut img = DynamicImage::new_rgba8(8, 8);
    for y in 0..8 {
        for x in 0..8 {
            let v = (x * 28 + y * 10) as u8;
            img.put_pixel(x, y, Rgba([v, 255u8.saturating_sub(v), 200, 255]));
        }
    }
    ok!("Canvas ready.");

    // Choose output filename
    let out_path = if let Some(ext) = input.extension().and_then(OsStr::to_str) {
        input.with_extension(format!("{ext}.png"))
    } else {
        input.with_extension("png")
    };

    step!("Writing PNG to {:?}…", &out_path.file_name().unwrap_or_default());
    // `png-achunk`'s Encoder implements `image::ImageEncoder` for image 0.24.x.
    img.write_with_encoder(
        achunk::Encoder::new_to_file(&out_path)
            .context("encoder")?
            .with_custom_chunk(chunk.clone()),
    )
    .context("writing PNG with custom chunk")?;
    ok!("PNG written.");

    // Re-open and verify immediately
    step!("Re-opening PNG to verify chunk & data…");
    let chunks = achunk::Decoder::from_file(&out_path)
        .context("decoder")?
        .decode_ancillary_chunks()
        .context("decode ancillary")?;
    let Some(found) = chunks
        .into_iter()
        .find(|c| c.chunk_type.to_ascii().as_str() == CHUNK_NAME)
    else {
        bail!("verification failed: missing chunk after write");
    };
    ok!(
        "Found chunk \"{}\" ({} bytes). CRC verified by decoder.",
        CHUNK_NAME,
        found.data.len()
    );

    step!("Parsing header back & verifying payload…");
    let (hdr2, rest2) = split_header_and_data(&found.data)?;
    hdr2.sanity_check()?;
    let verified = verify_bytes(hdr2.orig_len, hdr2.blake3, UnverifiedBytes(rest2.to_vec()))?;
    if verified.as_slice() != bytes.as_slice() {
        bail!("post-write byte-for-byte comparison failed");
    }
    ok!("Payload verified (length + BLAKE3 + exact bytes).");

    eprintln!();
    ok!("ENCODE COMPLETE → {:?}", out_path.file_name().unwrap_or_default());
    Ok(out_path)
}

fn decode_png_to_file(input: &Path) -> Result<PathBuf> {
    step!("Decoding PNG {:?} → original file…", input.file_name().unwrap_or_default());

    // Prove it's viewable by decoding with the `image` crate.
    step!("Checking PNG decodes as an image…");
    let _img = image::open(input).context("image::open PNG failed (not viewable?)")?;
    ok!("PNG successfully decoded (viewable).");

    // Extract ancillary chunks and find ours.
    step!("Reading ancillary chunks…");
    let chunks = achunk::Decoder::from_file(input)
        .context("decoder")?
        .decode_ancillary_chunks()
        .context("decode ancillary")?;
    let Some(found) = chunks
        .into_iter()
        .find(|c| c.chunk_type.to_ascii().as_str() == CHUNK_NAME)
    else {
        bail!("No embedded data chunk \"{}\" found; not created by this tool.", CHUNK_NAME);
    };
    ok!("Found chunk ({} bytes). CRC verified.", found.data.len());

    // Parse header & verify payload
    step!("Parsing header & verifying payload…");
    let (hdr, rest) = split_header_and_data(&found.data)?;
    hdr.sanity_check()?;
    let verified = verify_bytes(hdr.orig_len, hdr.blake3, UnverifiedBytes(rest.to_vec()))?;
    ok!(
        "Header OK. len={} hash={}",
        hdr.orig_len,
        hex32(hdr.blake3)
    );

    // Decide output path
    let desired_name = if hdr.orig_filename.trim().is_empty() {
        "decoded.bin".to_string()
    } else {
        hdr.orig_filename.clone()
    };
    let mut desired_path = input.with_file_name(&desired_name);

    // If desired file exists, but *already matches* expected length/hash, keep it.
    if desired_path.exists() {
        let (h, l) = blake3_hash_file(&desired_path)?;
        if l == hdr.orig_len && h == hdr.blake3 {
            ok!(
                "Existing file {:?} already matches embedded data; nothing to write.",
                desired_path.file_name().unwrap_or_default()
            );
            eprintln!();
            ok!("DECODE COMPLETE → {:?}", desired_path.file_name().unwrap_or_default());
            return Ok(desired_path);
        }
        // Else, avoid overwrite: append .restored
        warn!("{:?} exists with different content; appending .restored", desired_path.file_name().unwrap_or_default());
        desired_path = input.with_file_name(format!("{}.restored", desired_name));
    }

    // Write and re-verify
    step!("Writing reconstructed file to {:?}…", desired_path.file_name().unwrap_or_default());
    fs::write(&desired_path, verified.as_slice()).with_context(|| format!("write {:?}", desired_path))?;
    ok!("File written.");

    step!("Re-reading written file to re-verify…");
    let (rehash, relen) = blake3_hash_file(&desired_path)?;
    if relen != hdr.orig_len {
        bail!("output length mismatch after write ({} vs {})", relen, hdr.orig_len);
    }
    if rehash != hdr.blake3 {
        bail!("output BLAKE3 mismatch after write");
    }
    ok!("Output verified (length + BLAKE3).");

    eprintln!();
    ok!("DECODE COMPLETE → {:?}", desired_path.file_name().unwrap_or_default());
    Ok(desired_path)
}

// -------------------- entry --------------------

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
    if args.next().is_some() {
        bail!("exactly one argument is required");
    }
    let path = PathBuf::from(path_os);
    if !path.exists() {
        bail!("path does not exist: {:?}", path);
    }

    if is_png_file(&path)? {
        decode_png_to_file(&path)?;
    } else {
        encode_file_to_png(&path)?;
    }
    Ok(())
}

// -------------------- tests --------------------

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

    // ---------- Unit-ish tests ----------
    #[test]
    fn header_roundtrip_postcard() -> Result<()> {
        let hdr = PayloadHeader::new("f.bin".into(), 1234, [7u8; 32]);
        let bytes = postcard::to_allocvec(&hdr)?;
        let (back, rest) = postcard::take_from_bytes::<PayloadHeader>(&bytes)?;
        assert_eq!(back.orig_len, 1234);
        assert_eq!(rest.len(), 0);
        back.sanity_check()?;
        Ok(())
    }

    #[test]
    fn is_png_signature_check_works() -> Result<()> {
        let dir = tempdir()?;
        let png = dir.path().join("sigcheck.png");
        let mut img = DynamicImage::new_rgba8(2, 2);
        img.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        img.save(&png)?;
        assert!(is_png_file(&png)?);

        let not_png = dir.path().join("file.txt");
        fs::write(&not_png, b"not a png")?;
        assert!(!is_png_file(&not_png)?);
        Ok(())
    }

    // ---------- Integration-style tests (touch disk) ----------
    #[test]
    fn roundtrip_small_text_file() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("sample.txt");
        fs::write(&input, b"Hello ancillaries!\nThis is a test.\n")?;

        let png = encode_file_to_png(&input)?;
        assert!(png.exists());
        assert!(is_png_file(&png)?);

        let out = decode_png_to_file(&png)?;
        assert!(out.exists());

        let orig = fs::read(&input)?;
        let back = fs::read(&out)?;
        assert_eq!(orig, back);
        Ok(())
    }

    #[test]
    fn roundtrip_large_random_binary() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("blob.bin");
        // 256 KiB pseudo-random binary
        let data = gen_bytes(256 * 1024, 0xBAD5EED);
        fs::write(&input, &data)?;

        let png = encode_file_to_png(&input)?;
        let out = decode_png_to_file(&png)?;

        let back = fs::read(&out)?;
        assert_eq!(data.len(), back.len(), "length mismatch");
        assert_eq!(blake3_hash_bytes(&data), blake3_hash_bytes(&back), "hash mismatch");
        assert_eq!(data, back, "bytes mismatch");
        Ok(())
    }

    #[test]
    fn decode_plain_png_fails() -> Result<()> {
        let dir = tempdir()?;
        let png = dir.path().join("plain.png");

        // Write a plain PNG without our chunk.
        let mut img = DynamicImage::new_rgba8(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
        img.save(&png)?;

        let err = decode_png_to_file(&png).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("No embedded data chunk"), "unexpected: {msg}");
        Ok(())
    }

    #[test]
    fn decode_filename_conflict_appends_restored() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("report.dat");
        fs::write(&input, b"EDGE CASE FILE")?;

        // Encode → PNG
        let png = encode_file_to_png(&input)?;
        // Pre-create a file with the expected decoded name but *different content*
        let conflict = png.with_file_name("report.dat");
        fs::write(&conflict, b"pre-existing different")?;

        // Decode → should produce "report.dat.restored"
        let out = decode_png_to_file(&png)?;
        assert!(out.file_name().unwrap().to_str().unwrap().ends_with(".restored"));
        let back = fs::read(&out)?;
        assert_eq!(back, b"EDGE CASE FILE");
        Ok(())
    }

    #[test]
    fn filename_preserved_on_decode_when_same_file_already_present() -> Result<()> {
        // If the file already exists *and* matches len/hash, we keep the original name (no .restored).
        let dir = tempdir()?;
        let input = dir.path().join("very_long_filename.with.dots.and-spaces .bin");
        fs::write(&input, b"CONTENT")?;
        let png = encode_file_to_png(&input)?;

        // Now decode right next to the original—should detect same hash/len and NOT write ".restored".
        let out = decode_png_to_file(&png)?;
        assert_eq!(
            out.file_name().unwrap().to_str().unwrap(),
            "very_long_filename.with.dots.and-spaces .bin"
        );
        Ok(())
    }

    #[test]
    fn tamper_detection_via_hash_mismatch() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("secret.bin");
        fs::write(&input, b"THIS IS TOP SECRET")?;

        // Encode → PNG
        let png = encode_file_to_png(&input)?;

        // Read back image + chunks fully
        let (img, chunks) = achunk::Decoder::from_file(&png)
            .context("decoder")?
            .decode_all()
            .context("decode_all")?;

        // Find our chunk, parse header, flip a bit in payload
        let my = chunks
            .into_iter()
            .find(|c| c.chunk_type.to_ascii().as_str() == CHUNK_NAME)
            .expect("chunk not found");
        let (hdr, data) = super::split_header_and_data(&my.data)?;
        hdr.sanity_check()?;

        let mut tampered_payload = data.to_vec();
        if !tampered_payload.is_empty() {
            tampered_payload[0] ^= 0b0000_0001; // flip a bit
        } else {
            tampered_payload.extend_from_slice(b"x");
        }

        // Rebuild chunk = header || tampered_payload
        let mut rebuilt = postcard::to_allocvec(&hdr)?;
        rebuilt.extend_from_slice(&tampered_payload);

        let ctype = achunk::ChunkType::from_ascii(&CHUNK_NAME)?;
        let tampered_chunk = achunk::Chunk::new(ctype, rebuilt)?;

        // Write a *new* PNG with the tampered chunk
        let tampered_png = png.with_file_name("tampered.png");
        img.write_with_encoder(
            achunk::Encoder::new_to_file(&tampered_png)?.with_custom_chunk(tampered_chunk),
        )?;

        // Now decoding should fail due to hash mismatch
        let err = super::decode_png_to_file(&tampered_png).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("hash mismatch"), "unexpected error: {msg}");
        Ok(())
    }

    #[test]
    fn decode_idempotent_second_run_detects_existing_matching_file() -> Result<()> {
        let dir = tempdir()?;
        let input = dir.path().join("artifact.bin");
        fs::write(&input, b"XYZ")?;
        let png = encode_file_to_png(&input)?;

        // First decode
        let out1 = decode_png_to_file(&png)?;
        assert_eq!(out1.file_name().unwrap().to_str().unwrap(), "artifact.bin");

        // Second decode should detect existing matching file and return same path (no ".restored")
        let out2 = decode_png_to_file(&png)?;
        assert_eq!(out2.file_name().unwrap().to_str().unwrap(), "artifact.bin");
        Ok(())
    }
}
