import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

const phosphor = "#39ff14";
const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));
const snap = (n: number) => Math.round(n);

function seededShuffle<T>(arr: T[], seed: number) {
  const a = [...arr];
  let s = (seed || 1) >>> 0;
  for (let i = a.length - 1; i > 0; i--) {
    s = (1664525 * s + 1013904223) >>> 0;
    const j = s % (i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const PRESETS = {
  Unknown: { label: "Unknown Artifact", head: "?" as const, condDims: [768, 1024, 1280, 1536, 2048, 4096], latentC: [4, 8], scale: [8, 16], vision: ["ViT-L/14", "ViT-H/14", "SigLIP-H"] },
  SDXL:    { label: "SDXL-like",       head: "epsilon" as const, condDims: [2048], latentC: [4], scale: [8], vision: ["ViT-L/14"] },
  FLUX:    { label: "FLUX-like",       head: "v" as const,       condDims: [1536], latentC: [8], scale: [8], vision: ["ViT-H/14"] },
  WAN:     { label: "WAN21 Lightspeed",head: "epsilon" as const, condDims: [2048], latentC: [4], scale: [8], vision: ["ViT-H/14 (proj)"] },
  LLM:     { label: "LLM (GGUF)",      head: "tokens" as const,  condDims: [], latentC: [], scale: [], vision: [] },
} as const;

type PresetKey = keyof typeof PRESETS;
type HeadType = (typeof PRESETS)[PresetKey]["head"];
type LinkPoint = { x: number; y: number };
type Route = "HV" | "VH";

function estimatePerFrame({ width, height, steps, head, latentC }: { width: number; height: number; steps: number; head: HeadType; latentC: number }) {
  const baseStep = head === "epsilon" ? 0.16 : head === "v" ? 0.19 : head === "flow" ? 0.22 : head === "tokens" ? 0.10 : 0.16;
  const pxRef = 480 * 480;
  const resFactor = Math.max(1, width * height) / pxRef;
  const cFactor = (latentC || 4) / 4;
  return steps * baseStep * resFactor * cFactor;
}

/* ── Primitives ─────────────────────────────────────────────────────────── */

const Segment = React.memo(function Segment({ x, y, w, h, glow = phosphor }: { x: number; y: number; w: number; h: number; glow?: string }) {
  return (
    <div className="absolute" style={{ left: snap(x), top: snap(y), width: Math.max(2, w), height: Math.max(2, h) }}>
      <div className="w-full h-full" style={{ backgroundColor: glow, boxShadow: `0 0 8px ${glow}88`, opacity: 0.9 }} />
    </div>
  );
});

const OrthLink = React.memo(function OrthLink({ from, to, color = phosphor, route = "HV", active = true }: { from: LinkPoint; to: LinkPoint; color?: string; route?: Route; active?: boolean }) {
  if (!active) return null;
  const x1 = snap(from.x), y1 = snap(from.y), x2 = snap(to.x), y2 = snap(to.y);
  const hx = route === "HV" ? x2 : x1;
  const hy = route === "HV" ? y1 : y2;
  const segs = [
    { x: Math.min(x1, hx), y: y1 - 1, w: Math.abs(hx - x1), h: 2 },
    { x: hx - 1, y: Math.min(y1, hy), w: 2, h: Math.abs(hy - y1) },
    { x: Math.min(hx, x2), y: hy - 1, w: Math.abs(x2 - hx), h: 2 },
  ];
  return <>{segs.map((s, i) => <Segment key={i} {...s} glow={color} />)}</>;
});

const CRTNode = React.memo(function CRTNode({ title, subtitle, x, y, active }: { title: string; subtitle?: string; x: number; y: number; active: boolean }) {
  return (
    <div className={`absolute -translate-x-1/2 -translate-y-1/2 transition-opacity ${active ? "opacity-100" : "opacity-0"}`} style={{ left: snap(x), top: snap(y) }}>
      <div className="relative border rounded-lg px-3 py-2 min-w-[160px]"
           style={{ borderColor: `${phosphor}cc`, boxShadow: `0 0 10px ${phosphor}55, inset 0 0 18px ${phosphor}22`, background: "rgba(5,7,12,0.2)" }}>
        <div className="text-[10px] uppercase tracking-[0.2em]" style={{ color: phosphor }}>{title}</div>
        {subtitle && <div className="mt-1 text-[11px] text-emerald-200/90 font-mono">{subtitle}</div>}
        <div className="pointer-events-none absolute -inset-1">
          {["-left-1 -top-1 border-t border-l", "-right-1 -top-1 border-t border-r",
            "-left-1 -bottom-1 border-b border-l", "-right-1 -bottom-1 border-b border-r"]
            .map((cls, i) => <div key={i} className={`absolute w-3 h-3 ${cls}`} style={{ borderColor: phosphor }} />)}
        </div>
      </div>
    </div>
  );
});

const TimeBar = React.memo(function TimeBar({ value, max }: { value: number; max: number }) {
  const pct = clamp((value / Math.max(1e-6, max)) * 100, 0, 100);
  return (
    <div className="w-full h-2 bg-black border border-[rgba(57,255,20,0.35)] rounded">
      <div className="h-2 rounded" style={{ width: `${pct}%`, background: phosphor, boxShadow: `0 0 12px ${phosphor}` }} />
    </div>
  );
});

/* ── Main ───────────────────────────────────────────────────────────────── */

enum Phase { Idle=0, Cond=1, Head=2, Latent=3, Vision=4, Validate=5 }

export default function NeonProbeMockup() {
  const [preset, setPreset] = useState<PresetKey>("Unknown");
  const [phase, setPhase] = useState<Phase>(Phase.Idle);
  const [seed, setSeed] = useState<number>(1);
  const [probing, setProbing] = useState(false);
  const [res, setRes] = useState({ w: 854, h: 480 });
  const [steps, setSteps] = useState(20);
  const cfg = PRESETS[preset];

  const canvasRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ w: 0, h: 0 });

  useLayoutEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const measure = () => setCanvasSize({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!probing) return;
    if (phase >= Phase.Validate) { const t = setTimeout(() => setProbing(false), 300); return () => clearTimeout(t); }
    const t = setTimeout(() => setPhase(p => (p + 1) as Phase), 520);
    return () => clearTimeout(t);
  }, [probing, phase]);

  const discovered = useMemo(() => ({
    cond:   phase >= Phase.Cond     ? (cfg.condDims[0] ?? 2048) : undefined,
    head:   phase >= Phase.Head     ? (cfg.head !== "?" ? cfg.head : "epsilon") : undefined,
    latentC:phase >= Phase.Latent   ? (cfg.latentC[0] ?? 4) : undefined,
    scale:  phase >= Phase.Latent   ? (cfg.scale[0] ?? 8) : undefined,
    vision: phase >= Phase.Vision   ? (cfg.vision[0] ?? "ViT-L/14") : undefined,
    bound:  phase >= Phase.Validate
  }), [phase, cfg]);

  const perFrame = useMemo(() => {
    if (!discovered.head || !discovered.latentC) return 0;
    return estimatePerFrame({ width: res.w, height: res.h, steps, head: discovered.head as HeadType, latentC: discovered.latentC });
  }, [res, steps, discovered]);

  const W = canvasSize.w || 980, H = canvasSize.h || 660, cx = W / 2, cy = H / 2;

  const slots = useMemo(() => {
    const ring = Math.max(120, Math.min(W, H) / 2 - 120);
    const positions = [
      { x: cx - ring, y: cy - ring }, { x: cx, y: cy - ring }, { x: cx + ring, y: cy - ring },
      { x: cx + ring, y: cy }, { x: cx + ring, y: cy + ring }, { x: cx, y: cy + ring },
      { x: cx - ring, y: cy + ring }, { x: cx - ring, y: cy },
    ];
    return seededShuffle(positions, seed);
  }, [seed, cx, cy, W, H]);

  const nodeDefs = useMemo(() => ([
    { key: "text",    title: "TEXT ENCODER",      subtitle: discovered.cond ? `dim=${discovered.cond}` : "dim=", phase: Phase.Cond },
    { key: "model",   title: "MODEL / HEAD",      subtitle: discovered.head || "epsilon|v|flow", phase: Phase.Head },
    { key: "vae",     title: "VAE (LATENT LAW)",  subtitle: phase >= Phase.Latent ? `C=${discovered.latentC} · scale=${discovered.scale}×` : "C∈{4,8} · scale∈{8,16}", phase: Phase.Latent },
    { key: "vision",  title: "VISION ENCODER",    subtitle: discovered.vision || "ViT-L/14 | ViT-H/14 | SigLIP-H", phase: Phase.Vision },
    { key: "sampler", title: "K SAMPLER",         subtitle: phase >= Phase.Validate ? "1-step smoke: PASS" : "awaiting", phase: Phase.Validate },
  ].filter(n => phase >= n.phase)), [phase, discovered]);

  const positioned = useMemo(() => nodeDefs.map((n, i) => ({ ...n, ...slots[i] })), [nodeDefs, slots]);
  const center = { x: cx, y: cy };

  const transcript = useMemo(() => {
    const out: string[] = [];
    out.push("$ probe mystery_model.safetensors\n");
    out.push("[+] Format: safetensors   Size: 2.4 GB   Hash: 91f2…aa7c\n");
    out.push("[•] Sandbox 64×64 · steps=1 · timeout=250ms\n\n");
    out.push("┌ Handshake ───────────────────────────────┐\n");
    out.push(phase >= Phase.Cond    ? `  ✓ TEXT_EMB width: ${discovered.cond}\n` : "  … TEXT_EMB {768,1024,1280,1536,2048,4096}\n");
    out.push(phase >= Phase.Head    ? `  ✓ Sampler head: ${discovered.head}\n` : "  … sampler {epsilon,v,flow}\n");
    out.push(phase >= Phase.Latent  ? `  ✓ Latent law: C=${discovered.latentC} · scale=${discovered.scale}\n` : "  … latent C∈{4,8} scale∈{8,16}\n");
    out.push(phase >= Phase.Vision  ? `  ✓ Vision adapter: ${discovered.vision}\n` : "  … vision {ViT-L/14,ViT-H/14,SigLIP-H}\n");
    out.push(phase >= Phase.Validate? "  ✓ 1-step smoke: PASS (tensor→pixels)\n" : "  … 1-step smoke\n");
    out.push("└───────────────────────────────────────────\n\n");
    out.push("┌ Minimum skeleton ────────────────────────┐\n");
    if (discovered.bound) {
      out.push(`  string → TEXT_ENCODER(${discovered.cond}) → KSampler(${discovered.head}) → VAE.Decode → PIXELS\n`);
      out.push(`  PIXELS → VAE.Encode(C=${discovered.latentC},scale=${discovered.scale}) → (latent) → KSampler\n`);
    } else out.push("  (waiting for handshake…)\n");
    out.push("└───────────────────────────────────────────\n\n");
    out.push("┌ Cycle estimates (s/frame) ───────────────┐\n");
    const cases = [{ l: "480p · 20", w: 854, h: 480, s: 20 }, { l: "480p · 30", w: 854, h: 480, s: 30 }, { l: "720p · 20", w: 1280, h: 720, s: 20 }, { l: "720p · 30", w: 1280, h: 720, s: 30 }];
    cases.forEach(c => {
      const t = discovered.head && discovered.latentC ? estimatePerFrame({ width: c.w, height: c.h, steps: c.s, head: discovered.head as HeadType, latentC: discovered.latentC }) : 0;
      out.push(`  ${c.l.padEnd(9)} → ${t ? t.toFixed(2) : "—"}\n`);
    });
    out.push("└───────────────────────────────────────────\n\n");
    out.push("Obligations: ");
    if (!discovered.bound) {
      if (phase < Phase.Cond)   out.push("TEXT_ENCODER(dim=?); ");
      if (phase < Phase.Latent) out.push("VAE(latent_C=?,scale=?); ");
    } else out.push("(none)\n");
    return out.join("");
  }, [phase, discovered]);

  const maxDemo = useMemo(() => {
    const demo = estimatePerFrame({ width: 1280, height: 720, steps: 30, head: (discovered.head as HeadType) || "epsilon", latentC: discovered.latentC || 4 });
    return Math.max(demo, perFrame);
  }, [perFrame, discovered]);

  return (
    <div className="h-screen w-full bg-black text-emerald-100 overflow-hidden">
      <div className="h-full grid grid-cols-[3fr_2fr]">
        {/* Graph */}
        <div ref={canvasRef} className="relative">
          <div className="absolute inset-0 pointer-events-none" style={{ backgroundImage: "repeating-linear-gradient(0deg, rgba(57,255,20,0.05) 0, rgba(57,255,20,0.05) 1px, transparent 2px)", mixBlendMode: "screen", opacity: 0.35 }} />
          <div className="absolute inset-4 rounded-2xl border" style={{ borderColor: `${phosphor}55`, boxShadow: `0 0 40px ${phosphor}22` }} />
          <CRTNode title="MYSTERY MODEL" subtitle={PRESETS[preset].label} x={cx} y={cy} active />
          {positioned.map(n => <CRTNode key={n.key} title={n.title} subtitle={n.subtitle} x={n.x} y={n.y} active />)}
          {positioned.map((n, i) => {
            const route: Route = Math.abs(n.y - cy) > Math.abs(n.x - cx) ? "VH" : "HV";
            return <OrthLink key={i} from={center} to={{ x: n.x, y: n.y }} color={phosphor} route={route} active />;
          })}
          {/* Controls */}
          <div className="absolute left-6 top-6 flex items-center gap-3 font-mono text-xs">
            <span className="text-emerald-300/80">Preset</span>
            <select aria-label="Preset" className="bg-black/60 border border-[rgba(57,255,20,0.45)] rounded px-2 py-1"
              value={preset} onChange={(e) => { setPreset(e.target.value as PresetKey); setPhase(Phase.Idle); }}>
              {Object.keys(PRESETS).map(k => <option key={k} value={k}>{PRESETS[k as PresetKey].label}</option>)}
            </select>
            <button aria-pressed={probing} className={`px-3 py-1 rounded border ${probing ? "opacity-60" : ""}`}
              style={{ borderColor: phosphor, color: phosphor }}
              onClick={() => { setPhase(Phase.Idle); setSeed(s => (s + 1) >>> 0); setProbing(true); }}>PROBE</button>
            <button className="px-3 py-1 rounded border text-emerald-300" style={{ borderColor: `${phosphor}55` }}
              onClick={() => { setProbing(false); setPhase(Phase.Idle); }}>RESET</button>
          </div>
          <div className="absolute left-6 bottom-6 flex items-center gap-6 font-mono text-xs">
            <div>
              <div className="text-emerald-300/80 mb-1">Resolution</div>
              <div className="flex items-center gap-2">
                <button className={`px-2 py-1 rounded border ${res.w === 854 ? "border-[rgba(57,255,20,0.9)] text-emerald-200" : "border-[rgba(57,255,20,0.35)] text-emerald-300"}`} onClick={() => setRes({ w: 854, h: 480 })}>480p</button>
                <button className={`px-2 py-1 rounded border ${res.w === 1280 ? "border-[rgba(57,255,20,0.9)] text-emerald-200" : "border-[rgba(57,255,20,0.35)] text-emerald-300"}`} onClick={() => setRes({ w: 1280, h: 720 })}>720p</button>
              </div>
            </div>
            <div>
              <div className="text-emerald-300/80 mb-1">Steps</div>
              <div className="flex items-center gap-2">
                {[20, 30, 4].map(s => (
                  <button key={s} className={`px-2 py-1 rounded border ${steps === s ? "border-[rgba(57,255,20,0.9)] text-emerald-200" : "border-[rgba(57,255,20,0.35)] text-emerald-300"}`} onClick={() => setSteps(s)}>{s}</button>
                ))}
              </div>
            </div>
            <div className="min-w-[280px] text-emerald-300">
              <div>Est. time: <span className="text-emerald-200">{perFrame ? perFrame.toFixed(2) : "—"} s/frame</span></div>
              <TimeBar value={perFrame} max={maxDemo} />
            </div>
          </div>
        </div>
        {/* Right panel */}
        <div className="relative border-l" style={{ borderColor: `${phosphor}33` }}>
          <div className="absolute inset-0 pointer-events-none" style={{ backgroundImage: "repeating-linear-gradient(0deg, rgba(57,255,20,0.05) 0, rgba(57,255,20,0.05) 1px, transparent 2px)", opacity: 0.4 }} />
          <div className="p-4 font-mono text-sm">
            <div className="text-emerald-300 tracking-widest text-xs uppercase">Analysis</div>
            <pre role="log" aria-live="polite" className="mt-2 whitespace-pre-wrap text-emerald-200/90 leading-relaxed">{transcript}</pre>
            <div className="mt-4 grid grid-cols-2 gap-3 text-[12px]">
              <div className="border rounded p-2" style={{ borderColor: `${phosphor}55` }}>
                <div className="text-emerald-300/90">Ports</div>
                <ul className="mt-1 space-y-1">
                  <li>TEXT_EMB: {discovered.cond ?? "?"}</li>
                  <li>HEAD: {discovered.head ?? "?"}</li>
                  <li>LATENT: C={discovered.latentC ?? "?"} · scale={discovered.scale ?? "?"}</li>
                  <li>VISION: {discovered.vision ?? "?"}</li>
                </ul>
              </div>
              <div className="border rounded p-2" style={{ borderColor: `${phosphor}55` }}>
                <div className="text-emerald-300/90">Cycle (s/frame)</div>
                {[{ l: "480p·20", w: 854, h: 480, s: 20 }, { l: "480p·30", w: 854, h: 480, s: 30 }, { l: "720p·20", w: 1280, h: 720, s: 20 }, { l: "720p·30", w: 1280, h: 720, s: 30 }].map((c, i) => {
                  const t = discovered.head && discovered.latentC ? estimatePerFrame({ width: c.w, height: c.h, steps: c.s, head: discovered.head as HeadType, latentC: discovered.latentC }) : 0;
                  return (
                    <div key={i} className="flex items-center gap-2">
                      <div className="w-20">{c.l}</div>
                      <div className="flex-1"><TimeBar value={t} max={maxDemo} /></div>
                      <div className="w-14 text-right">{t ? t.toFixed(2) : "—"}</div>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="mt-3 text-emerald-300/80 text-xs">Preset → PROBE to reshuffle layout; diagram honors 90° links and space-fits around the center capsule.</div>
          </div>
        </div>
      </div>
    </div>
  );
}

