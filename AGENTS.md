## UNIVERSAL GUIDANCE

## Prototype-Driven Development Compass

Codex must use this file as its constant reference for how to define and complete tasks.
Every task must produce forward progress by adding a new capability that is visible in
the prototype and mirrored in the main project.

---

### Core Rules

1. **Every task adds a capability.**  
   - Always create or extend a callable in the core project (function or class).  
   - Add a corresponding probe in `prototype.py` that calls this callable.  
   - Prototype never invents standalone logic; it always mirrors what exists in core code.

2. **Green lights prove real work.**  
   - A probe is complete only if it performs an observable action: writes a file, makes
     an HTTP call, runs a CUDA op, or logs structured data.  
   - Hardcoded success messages are not valid. Success must come from actual IO or compute.

3. **Prototype is the changelog.**  
   - Each probe prints `[OK] name(): description` when it succeeds.  
   - The description is one line explaining what capability was added and why it matters.  
   - Append structured metadata for each probe to `_artifacts/proto.json`.

4. **Stable facades, flexible internals.**  
   - Always expose a thin public facade in the core project for the prototype to call.  
   - Internals may change, but the facade and its prototype probe remain stable.

5. **Bounded, observable probes.**  
   - Each probe must run quickly (a few seconds).  
   - Each probe must emit an artifact in `_artifacts/` (file, log, JSON, response).  
   - All probes must run offline or against localhost only. No external network.  
   - If unsupported in the environment, print `[SKIP]` with a clear reason.

6. **Behavioral invariants.**  
   - Probes check properties of outputs, not exact strings.  
   - Examples: file is nonempty, audio has correct sample rate, HTTP returns status 200.  
   - Never design brittle checks that break on valid variations.

7. **No regressions.**  
   - Once a green light exists, it must remain.  
   - If internals change, update the prototype probe so the light stays green.  
   - Never remove a light without adding an equivalent or stronger one.

8. **Always human-readable.**  
   - Every green light must include a one-line rationale.  
   - This text is the running release note and project log.

9. **Shadow comparisons for optimizations.**  
   - When adding an accelerated or alternative path, run both baseline and new path.  
   - Log both results and print the comparison in the prototype output.  
   - Example: `[OK] accel_fp4(): baseline=120ms new=85ms speedup=1.4x`.

10. **Definition of Done.**  
    - A task is complete only if `prototype.py` produces one more `[OK]` line than before.  
    - That `[OK]` must come from a real callable in the core project and a working probe
      in the prototype.  
    - If no new `[OK]` appears, the task is not complete.

---

### Summary

- Prototype drives progress: every capability is mirrored in both core and prototype.  
- Success is proven by observable IO, not print statements.  
- Each probe doubles as a changelog entry and performance log.  
- Codex cannot progress by editing docs, dependencies, or tests alone; only a new green light counts.  
- Over time, the prototype becomes a running ignition checklist of everything the project can do.



# OPERATING MODEL: AUMLIT

**Goal:** Given an unknown model artifact (`.safetensors`, `.gguf`, `.onnx`), emit:

1. a **contact trace** (minimum runnable workflow skeleton),
2. a set of **obligations** (missing components with types),
3. a **proof transcript** (tried candidates, failures with extracted integers, successful binding, and micro-perf).

**Method:** Run **tiny, bounded probes** against **real execution oracles** (PyTorch, xFormers, llama.cpp, ONNX Runtime). Use a **finite candidate space** and a **typed error classifier** to collapse uncertainty.

---

## Core loop (engine-agnostic pseudocode)

```
inputs: artifact_path
outputs: contact_trace.json, obligations.json, proof.txt

1. meta = read_headers(artifact_path)          # GGUF/ONNX metadata; safetensors headers if present
2. HYP = seed_hypotheses(meta)                 # tokenizer hints, rope, ctx, latent scale, etc.

3. CANDIDATES = plan_candidates(HYP, defaults={
       TEXT_EMB_d: [768,1024,1280,1536,2048,4096],
       HEAD:       ["epsilon","v","flow"],
       LATENT_C:   [4,8],
       LATENT_SCALE:[8,16],
       VISION:     ["ViT-L/14-grid","ViT-H/14-grid","SigLIP-H-map"]
   })

4. SANDBOX = spawn_sandbox(limits={timeout=2s, vram_cap=1.0GB, cpu_mem_cap=2GB})
   # never unpickle; only safetensors/gguf/onnx; isolate kernels; kill-on-timeout

5. while unresolved(HYP):
       probe = next_probe(CANDIDATES)          # cheap→expensive; max info gain
       puppet = craft_sock_puppet(probe)       # tiny tensors: images 64x64→latents 8x8; text 1–2 tokens
       try:
           result = SANDBOX.try_forward(artifact_path, puppet, engines=[PT, XFORMERS, LLAMA, ORT])
           HYP = update_hypotheses(HYP, result.accepted_bindings)
           record_success(probe, result)
       except EngineError as e:
           reason = CLASSIFIER.type(e)         # COND_DIM, LATENT_C, HEAD, VISION_ADAPTER, OTHER
           HYP = update_from_reason(HYP, reason.extracted_integers)
           record_failure(probe, reason)

       if satisfied_min_run(HYP) and validated_one_step(SANDBOX, HYP):
           break

6. TRACE = build_contact_trace(HYP, SANDBOX.validation_artifacts)
7. OBLIG = derive_obligations(HYP)             # “missing TEXT_ENCODER(dim=2048)”, etc.
8. PROOF = compile_proof_transcript()

9. emit(TRACE, OBLIG, PROOF)
```

---

## Components (clear interfaces)

### 1) Oracles (engines)

* `TorchOracle.try_forward(module, inputs) -> Accept|Raise(error)`
  *Contract:* runs a 1-step forward; **no dynamic code execution**; uses `safetensors` or known modules; collects first failing op kind + shapes.
* `XFormersOracle.try_attention(layout_hint) -> Accept|Raise(error)`
  *Contract:* toggles mem-efficient attention; failures imply layout/stride/type mismatches.
* `LlamaCppOracle.ping(gguf_file, token_id) -> Accept|Raise(error)`
  *Contract:* loads GGUF metadata; builds KV cache; 1–2 token step; errors reveal vocab, rope, kv dtypes.
* `OnnxOracle.infer(onnx_file, io_shapes_hint) -> Accept|Raise(error)`
  *Contract:* ORT shape inference; if dynamic axes remain, run with concrete tiny shapes.

**Return discipline:** when raising, include **op kind** and **relevant integers** (dim lengths, in/out channels, kernel stride) extracted from the runtime error.

---

### 2) Sock-puppets (dummy emitters)

* `TextEmb(d)` → tensor `[1, T=2, d]`
* `VisionGrid(profile)` → tensor shaped for `{ViT-L/14, ViT-H/14, SigLIP-H}`
* `Latent(C, scale)` → tensor `[1, C, H/scale, W/scale]` for base `H=W=64`
* `AudioMel(frame_shape)` (future)
* `KVCacheProbe(n_heads, d_head, T=2)` (LLM)

All puppets **guarantee geometry** (not semantics). Dtypes default to fp16/f32 with fallback.

---

### 3) Classifier (typed, tiny taxonomy)

Input: raw engine error (string/enum + attached tensor shapes)
Output: `{type, integers}` from:

* `COND_DIM` (e.g., “matmul (...×2048 vs ...×1536)” → `required_d=2048`)
* `LATENT_C` (“Conv2d expected in\_channels=4, got 8” → `required_C=4`)
* `HEAD` (“prediction mismatch: sampler ε, model v” → `required_head=v`)
* `VISION_ADAPTER` (“reshape/view fails at join” → `profile inference`)
* `KV/VOCAB/ROPE` (LLMs) (`vocab=32k`, `rope=128k`, `kv_dtype=f16`)
* `OTHER` (logged; doesn’t block unless persistent)

**Rule source:** regex + structured hooks; prioritize integers over phrasing.

---

### 4) Planner (probe ordering)

Heuristic: **fail fast, collapse big**
Order: `TEXT_EMB_d → HEAD → LATENT_C/scale → VISION profile → (LLM KV/ROPE specifics)`
Optional Bayesian policy: score candidates by expected information gain from past failures.

---

### 5) Sandbox (safety envelope)

* **Process isolation** (subprocess)
* **Resource caps** (CUDA memory fraction; ulimit; Windows Job Objects)
* **Timeouts** (per probe + global)
* **No pickles**; only `safetensors/gguf/onnx`
* **Deterministic seed** for repeatability
* **Hash cache**: `artifact_hash + probe_signature → outcome`

---

### 6) Printers (outputs)

* **Contact trace (JSON DAG)**
  Schema (see below). Also render ComfyUI subgraph and CLI recipe.
* **Obligations (JSON list)**
  Required external components with typed sockets.
* **Proof transcript (text)**
  Chronological attempts, extracted integers, bindings, `t_step@64²`, `vram@64²`.

---

## Data contracts (schemas)

### `contact_trace.json`

```json
{
  "nodes": [
    {"id":"input.text", "type":"string"},
    {"id":"text_emb", "type":"TEXT_EMB", "params":{"d":2048}},
    {"id":"vision", "type":"VISION_FEAT", "params":{"profile":"ViT-H/14-grid"}, "optional": true},
    {"id":"vae.enc", "type":"VAE_ENC", "params":{"C":4, "scale":8}},
    {"id":"model", "type":"UNET", "params":{"head":"epsilon"}},
    {"id":"vae.dec", "type":"VAE_DEC"},
    {"id":"output.image", "type":"PIXELS", "shape":{"H":64, "W":64, "C":3}}
  ],
  "edges": [
    ["input.text","text_emb"],
    ["text_emb","model"],
    ["vision","model"],
    ["vae.enc","model"],
    ["model","vae.dec"],
    ["vae.dec","output.image"]
  ],
  "evidence": {
    "validated": true,
    "t_step_ms_64": 12.4,
    "vram_mb_64": 410
  }
}
```

### `obligations.json`

```json
[
  {"need":"TEXT_ENCODER", "params":{"d":2048}},
  {"need":"VAE", "params":{"C":4, "scale":8}},
  {"need":"TOKENIZER", "params":{"family":"CLIP/T5"}, "confidence":0.7}
]
```

### `proof.txt` (free-form but structured)

```
artifact: 3f5e... (safetensors)
seed: 1234

meta: none
plan: d=[1536,2048], head=[epsilon,v], C=[4,8], scale=[8,16], vision=[H14,L14,SigLIP]

[1] probe TEXT_EMB d=1536 → FAIL
    error: matmul shapes cannot be multiplied (...×1536 vs ...×2048)
    classify: COND_DIM{required_d=2048}

[2] probe TEXT_EMB d=2048 + HEAD=epsilon + C=8 → FAIL
    error: Conv2d expected in_channels=4, got 8
    classify: LATENT_C{required_C=4}

[3] probe TEXT_EMB d=2048 + HEAD=epsilon + C=4 + scale=8 → OK
    validate one-step@64²: OK (12.4ms, 410MB)

contact: UNET(head=epsilon), VAE(C=4,scale=8), TEXT_EMB(d=2048), optional VISION
```

---

# Repo scaffold (drop-in for Codex)

```
universal-model-scanner/
├─ README.md
├─ pyproject.toml                      # or package.json if Node; language-agnostic here
├─ src/
│  ├─ cli/
│  │  └─ main.py                       # seed executable (see below)
│  ├─ core/
│  │  ├─ planner.py                    # plan_candidates, next_probe
│  │  ├─ sandbox.py                    # spawn_sandbox, resource caps, caches
│  │  ├─ classifier.py                 # typed error taxonomy, regex rules
│  │  ├─ printers.py                   # contact_trace, obligations, proof
│  │  ├─ schemas.py                    # JSON schema validators
│  │  └─ metadata.py                   # read_headers: gguf/onnx/safetensors
│  ├─ oracles/
│  │  ├─ pytorch_oracle.py             # TorchOracle
│  │  ├─ xformers_oracle.py            # XFormersOracle
│  │  ├─ llama_cpp_oracle.py           # LlamaCppOracle
│  │  └─ onnx_oracle.py                # OnnxOracle
│  ├─ puppets/
│  │  ├─ text_emb.py                   # TextEmb(d)
│  │  ├─ latent.py                     # Latent(C,scale)
│  │  ├─ vision.py                     # VisionGrid(profile)
│  │  └─ kvcache.py                    # KVCacheProbe(...)
│  ├─ adapters/                        # representation, not meaning
│  │  ├─ tokenizers/
│  │  │  ├─ clip.py
│  │  │  └─ t5.py
│  │  ├─ vae.py                        # encode/decode stubs with C/scale
│  │  ├─ pack.py                       # layout/dtype pack/unpack
│  │  └─ samplers.py                   # epsilon↔v↔flow family switch
│  └─ utils/
│     ├─ hashing.py                    # file hash + probe signature
│     ├─ logging.py
│     └─ timing.py
├─ rules/
│  └─ error_rules.yaml                 # regex → {type, pull integers}
├─ examples/
│  ├─ traces/                          # gold samples
│  └─ artifacts/README.md              # where to drop test ckpts
├─ tests/
│  ├─ unit/
│  │  ├─ test_classifier.py            # synthetic errors → correct type/ints
│  │  ├─ test_planner.py
│  │  └─ test_sandbox.py
│  └─ integration/
│     ├─ test_unet_probe.py            # fake modules that raise controlled errors
│     └─ test_llama_ping.py
```

> If you want this in another language, keep the structure and names; only mechanics change.

---

# Seed executable (minimal CLI behavior)

**`src/cli/main.py` (language-agnostic pseudocode)**

```python
def main():
    args = parse_args()  # --artifact PATH --out OUTDIR --max-time 10s --no-xformers --no-onnx, etc.
    meta = metadata.read_headers(args.artifact)
    hyp  = planner.seed_hypotheses(meta)
    with sandbox.spawn(args.limits) as box:
        proof = []
        plan  = planner.plan_candidates(hyp, args.overrides)
        cache = load_cache()

        for probe in planner.iter_probes(plan):
            if cache.hit(args.artifact, probe):
                outcome = cache.get(args.artifact, probe)
                proof.append(outcome)
                planner.update(hyp, outcome)
                if planner.ready(hyp): break
                continue

            try:
                outcome = box.try_probe(args.artifact, probe)
                classifier.note_success(outcome)
                planner.update(hyp, outcome.bindings)
            except EngineError as e:
                reason = classifier.type(e)
                outcome = record_failure(probe, reason)
                planner.update(hyp, reason.ints)

            cache.put(args.artifact, probe, outcome)
            proof.append(outcome)
            if planner.ready(hyp):
                validate = box.validate_min_run(args.artifact, hyp)
                proof.append(validate)
                break

    trace = printers.contact_trace(hyp, validate)
    oblig = printers.obligations(hyp)
    printers.proof(proof)

    write_json(args.out/"contact_trace.json", trace)
    write_json(args.out/"obligations.json", oblig)
    write_text(args.out/"proof.txt", "\n".join(proof))
```

**CLI contract:**

```
ums probe --artifact PATH [--out OUTDIR]
          [--guess "d=2048, C=4, head=epsilon"]    # optional overrides
          [--limits "timeout=2s,vram=1GB"]        # per-probe caps
          [--format comfy|json|cli]               # extra renderers
```

---

# Implementation notes (the levers Codex should pull)

* **Resource control:**

  * CUDA: set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:...` and per-process GPU memory fraction; for CPU mem use `ulimit`/Job Objects.
  * Kill on timeout with a watchdog thread/process; treat timeouts as `OTHER` unless reproducible.
* **Header reading:**

  * GGUF: vocab, rope, context, tensor names/shapes, quantization types.
  * ONNX: use shape inference; collect dynamic axes; fall back to concrete tiny shapes.
  * Safetensors: collect tensor keys & shapes; do **not** exec code; only map shapes.
* **xFormers toggle:** try mem-efficient attention first; on failure, record layout hint and fall back to vanilla attention.
* **Classifier robustness:** match on **numbers** first; keep phrasing patterns broad; store examples in `rules/error_rules.yaml` with golden tests.
* **Cache key:** `SHA256(artifact) + SHA256(probe_signature)`; store verdict + extracted integers + timings.
* **Contact trace to ComfyUI:** maintain a small mapping: node-type → port names + params; assert types (e.g., `TEXT_EMB[d] → CrossAttn.cond`).
* **Security:** never load Python pickles; gate dynamic import; for ONNX disable custom ops; for llama.cpp, load from file only.

---

# Multiple viable strategies (Codex can pick)

1. **Exception-mining first (recommended)**
   Treat engines as black-box oracles; mine exceptions for integers. Fastest to MVP; aligns with your concept.
2. **Hybrid shape propagation**
   Wrap forwards with hooks to capture **actual** intermediate shapes when possible; reduces reliance on string parsing; still uses exceptions to bind remaining vars.
3. **Constraint solver assist**
   Maintain a simple constraint set (dims, heads, scales) and use exceptions to add clauses; a tiny SAT pass prunes candidates aggressively.

I’d start with (1), add (2) opportunistically, and keep (3) as a thin layer for cleanliness (not required).

---

# Test plan (codify reality checks)

* **Unit:**

  * Classifier: >50 synthetic error strings → correct `{type, integers}`.
  * Planner: ensures `TEXT_EMB → HEAD → LATENT → VISION` ordering, with back-off.
  * Sandbox: enforces timeouts & caps; proves kill-on-limit.
* **Integration:**

  * Fake UNet that intentionally raises `in_channels≠C`; expect `LATENT_C` extraction.
  * Fake cross-attn with wrong cond dim; expect `COND_DIM`.
  * GGUF ping on a tiny test model; assert vocab/rope extraction.
  * ONNX model with dynamic axes; ORT inference reduces ambiguity or emits obligation.
* **Golden:**

  * Known SD-like safetensors → trace with `C=4, scale=8, head=epsilon, d=2048`.
  * Known Llama GGUF → trace with vocab/rope/kv dtypes.

---

# Example outputs (what “good” looks like)

**Trace (CLI render):**

```
string → TEXT_EMB[d=2048] → UNET(head=epsilon) ← VAE.Encode[C=4,scale=8]
UNET → VAE.Decode → PIXELS[64,64,3]
(optional) VISION_FEAT[ViT-H/14 grid] → UNET
```

**Obligations:**

* `TEXT_ENCODER(dim=2048)`
* `VAE(C=4, scale=8)`
* `TOKENIZER(family=CLIP/T5)` (confidence 0.7)

**Proof (excerpt):**

```
[2] XFORMERS mem_efficient=True → FAIL (layout mismatch); fallback used; layout=sdpa
[3] Torch: Conv2d expected in_channels=4, got 8 → LATENT_C{4}
[4] Torch: matmul (...×1536 vs ...×2048) → COND_DIM{2048}
[5] One-step@64² OK (12.4ms, 410MB)
```

---

