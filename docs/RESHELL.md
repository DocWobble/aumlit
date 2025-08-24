# Reshell Overview

`reshell` is a subcommand of the `aumlit` CLI that reconstructs the minimal
workflow needed to run an unknown model file.

## Foundational Libraries
- safetensors – read tensor weights without pickling risk.
- torch – primary engine for UNet/T2I probes.
- xformers – optional mem-efficient attention toggling.
- onnxruntime – run ONNX encoders/decoders.
- llama.cpp (CLI or bindings) – minimal GGUF LLM steps.
- numpy – craft dummy tensors for ONNX and generic ops.
- huggingface_hub – fetch model headers / metadata when available.

## MVP Sprint Plan
1. Skeleton CLI and minimal project setup.
2. Detect format & read headers.
3. Candidate generation.
4. Sock-puppets (dummy tensor emitters).
5. Brute-force probing loop.
6. Emit minimal ComfyUI workflow.
7. Record obligations & proof transcript.
8. Stretch goals: GGUF/LLM probing, simple planner, hardened sandbox, more node types.

## Vision
Drop an unknown model file into aumlit, run `aumlit reshell`, and receive a minimal
workflow mapping the model, listing required tools and the I/O contact points needed
to run it.

## Project Summary
The repository is laying out a safety-oriented probing engine for unknown model files. It runs tiny, bounded tests against real execution backends (PyTorch, xFormers, llama.cpp, ONNX Runtime) to automatically derive a minimal contact trace of how the artifact should run, list any missing components as obligations, and record a proof transcript of all probe results. Design principles emphasize "System structure flows deterministically from geometry" - strict loading policies (no pickles, resource caps, kill-on-timeout), and an initial scope around UNet-style T2I models, ControlNet variants, basic GGUF LLMs, and ONNX encoders/decoders.
