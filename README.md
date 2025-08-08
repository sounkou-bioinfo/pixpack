# pixpack

Turn **any file into a single PNG** whose **pixels carry the data**. Feed that PNG back in to **recover the exact original** with strong integrity checks.

# Install

```bash
cargo install pixpack
```

# Use

```bash
# encode (file → PNG)
pixpack myfile.bin        # creates myfile.bin.png

# decode (PNG → file)
pixpack myfile.bin.png    # recreates myfile.bin
```

One argument only. If it’s a PNG, pixpack decodes; otherwise it encodes.

# Info

* Renders a **macro-cell grid** with a **white quiet zone** and **black frame**.
* Packs: `MAGIC | VERSION | header (×2) | payload | trailer_u32`.

  * **Header**: filename, total length, full BLAKE3, grid hints.
  * **Trailer**: first 4 bytes of BLAKE3(payload) for a fast sanity check.
* **Decode**: grayscale → Otsu threshold (with a few fallbacks) → infer frame/quiet → sample **center pixel per cell** → rebuild stream.
* **Integrity**:

  * Encode: re-open the written PNG, decode it back, verify **length + full BLAKE3 + byte-for-byte**.
  * Decode: after writing the file, re-hash and verify **length + full BLAKE3** again.
* Capacity is **1 bit per data cell**; pixpack picks a near-square grid and a cell size capped by a max canvas side.
