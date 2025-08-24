* **Principle:** “System structure flows deterministically from geometry.”
* **Safety:** no pickles; strict loaders; per-probe caps; kill-on-timeout.
* **Scope:** start with UNet-like T2I, ControlNet variants, basic GGUF LLMs, ONNX enc/dec.
* **Non-goals (MVP):** full sampler zoo, training graphs, exotic MoE/flash-decoding specifics.
* **Extensibility:** taxonomy file, new puppet families, new oracle wrappers.


* **Docs:** see [docs/RESHELL.md](docs/RESHELL.md) for foundation libraries, MVP sprint plan, vision, and project summary.

## Installation

Install the toolkit from PyPI:

```bash
pip install aumlit
```

## Usage

After installation the `aumlit` command exposes the `reshell` subcommand:

```bash
aumlit reshell path/to/model.safetensors --out out_dir
```

The command prints placeholder locations for `contact_trace.json`, `obligations.json`, and `proof.txt`.

## Concurrency

Cache files such as `failure_cache.json` and `header_cache.json` are written
under a cross-platform file lock. This prevents concurrent `reshell` runs from
corrupting the JSON when multiple processes update the same cache directory.

## Building standalone binaries

Standalone binaries can be produced with [PyInstaller](https://pyinstaller.org/).

On Linux and macOS:

```bash
make build
```

On Windows:

```powershell
.\build.ps1
```

The resulting executable is written to the `dist/` directory.

Prebuilt binaries for common platforms are published with each GitHub release.
