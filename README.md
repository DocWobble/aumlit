* **Principle:** “Real engines, tiny probes, typed reasons.”
* **Safety:** no pickles; strict loaders; per-probe caps; kill-on-timeout.
* **Scope:** start with UNet-like T2I, ControlNet variants, basic GGUF LLMs, ONNX enc/dec.
* **Non-goals (MVP):** full sampler zoo, training graphs, exotic MoE/flash-decoding specifics.
* **Extensibility:** taxonomy file, new puppet families, new oracle wrappers.


* **Docs:** see [docs/RESHELL.md](docs/RESHELL.md) for foundation libraries, MVP sprint plan, vision, and project summary.

## Usage

Run the skeleton CLI:

```bash
python reshell.py --artifact path/to/model.safetensors --out out_dir
```

The command prints placeholder locations for `contact_trace.json` and `obligations.json`.
