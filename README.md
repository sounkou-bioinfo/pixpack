# pixpack

Pack **any file** into a **viewable PNG**, and unpack it back to the **exact same bytes** — in one command, with **cryptographic proof**.

* **One arg only:** if the path is a PNG (by signature) → **decode**; otherwise → **encode**.
* **Viewable output:** writes a tiny 8×8 gradient PNG; your data lives in a private ancillary chunk.
* **Integrity first:** stores length + **BLAKE3** in a header and **re-verifies** after every write.

---

## Install

```bash
cargo install pixpack
```

---

## Usage

**Encode (file → PNG):**

```bash
pixpack file.txt
```

**Decode (PNG → file):**

```bash
pixpack file.png
```

Behavior on conflicts:

* If `file.ext` **exists** and **matches** the embedded length+hash → **no write** (idempotent).
* If it **differs** → write `file.ext.restored`.

---

## How it works (short)

* Payload is stored in PNG **ancillary chunk** `ruSt`.
* Chunk data = **postcard** header (magic, version, original name, length, BLAKE3) + raw bytes.
* Encode path: write PNG → re-open → parse chunk → verify length/hash and **byte-for-byte equality**.
* Decode path: verify PNG decodes, parse chunk, verify, write file, **rehash** file.

---

## Guarantees

* **Lossless round-trip** (length + BLAKE3 match).
* **Actively proven** after every write.
* PNG remains **100% viewable** in normal image viewers.

---

## Caveats

* **Not encrypted/hidden.** Anyone can inspect chunks.
* Some tools/services **strip ancillary chunks** → don’t run optimized/stripping pipelines on encoded PNGs.
* Embedding currently loads payload into memory (very large files may not be ideal).

---

## Quick examples

```bash
# Pack a config
echo "data=42" > app.conf
pixpack app.conf            # -> app.conf.png
pixpack app.conf.png        # -> app.conf (same bytes)

# Idempotent decode
pixpack artifact.bin.png    # writes artifact.bin
pixpack artifact.bin.png    # detects match; no duplicate
```

---

That’s it — **pack** with confidence, **unpack** with proof.

(Co-written with GPT-5.)
