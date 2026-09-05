const pptx = require("pptxgenjs");
const p = new pptx();
p.layout = "LAYOUT_WIDE";                 // 13.3 x 7.5
p.author = "AHC Visual Intelligence Hackathon";
p.title = "Near-Real-Time Video Anomaly Detection";

// Palette: surveillance-dark with an amber alert accent.
const INK = "10151F";      // slide ground
const PANEL = "1B2436";    // raised card
const PANEL2 = "222E44";   // lighter card
const ICE = "CADCFC";      // body text
const MUTE = "8496B4";     // captions
const AMBER = "F2A63B";    // anomaly accent
const GREEN = "5FCB8E";    // wins
const RED = "E2685F";      // negatives
const WHITE = "FFFFFF";

const H = "Cambria";       // safe-list serif header
const B = "Calibri";       // safe-list sans body

const card = (s, x, y, w, h, fill) =>
  s.addShape(p.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06, fill: { color: fill || PANEL }, line: { color: fill || PANEL },
  });

// ────────────────────────────── SLIDE 1 ──────────────────────────────
const s1 = p.addSlide();
s1.background = { color: INK };

s1.addText("Small VLM-free cascade for near-real-time anomaly detection", {
  x: 0.55, y: 0.34, w: 12.2, h: 0.5, fontFace: H, fontSize: 27, bold: true, color: WHITE, margin: 0,
});
s1.addText("Frozen encoder + 2.2M-parameter temporal head. No model in the runtime path larger than SigLIP-base.", {
  x: 0.55, y: 0.87, w: 12.2, h: 0.32, fontFace: B, fontSize: 13, color: MUTE, margin: 0,
});

// ── pipeline
const px = [0.55, 3.02, 5.49, 7.96, 10.43], pw = 2.32, py = 1.42, ph = 1.16;
const steps = [
  ["Decode @ 2 fps", "grab/skip, resize 224", MUTE],
  ["SigLIP-base", "frozen, cached (T,768)", MUTE],
  ["BiGRU head", "2.18M params", AMBER],
  ["Hysteresis + merge", "few, long segments", MUTE],
  ["Events JSON", "class + interval", MUTE],
];
steps.forEach((st, i) => {
  card(s1, px[i], py, pw, ph, i === 2 ? PANEL2 : PANEL);
  s1.addText(st[0], { x: px[i] + 0.14, y: py + 0.19, w: pw - 0.28, h: 0.3,
    fontFace: B, fontSize: 13, bold: true, color: i === 2 ? AMBER : ICE, margin: 0 });
  s1.addText(st[1], { x: px[i] + 0.14, y: py + 0.56, w: pw - 0.28, h: 0.42,
    fontFace: B, fontSize: 10.5, color: MUTE, margin: 0 });
  if (i < 4) s1.addText("›", { x: px[i] + pw - 0.02, y: py + 0.34, w: 0.2, h: 0.4,
    fontFace: B, fontSize: 20, color: AMBER, align: "center", margin: 0 });
});

// ── three decisions
const dy = 2.92, dh = 1.62, dx = [0.55, 4.72, 8.89], dw = 3.86;
const dec = [
  ["Train data ≠ test data",
   "Every training clip is ~5 s and single-event. Levels 2–3 are 240–629 s and multi-event. We synthesise long sequences by concatenating clip embeddings, carrying offsets through to exact labels — free supervision for the regime we are scored on."],
  ["Dense labels, no MIL",
   "Audit showed all 2 200 anomaly rows already carry timestamps, so the planned MIL pooling was unnecessary. But 1 551 span their whole clip, so only ~649 teach boundaries — those get 3× loss weight."],
  ["Precision over recall",
   "Scoring punishes false alarms far harder than misses. A perfect event set split into five fragments drops L2 from 1.00 to 0.47. So: high enter threshold, 20 s merge gap, few long segments."],
];
dec.forEach((d, i) => {
  card(s1, dx[i], dy, dw, dh);
  s1.addShape(p.ShapeType.ellipse, { x: dx[i] + 0.2, y: dy + 0.2, w: 0.3, h: 0.3,
    fill: { color: AMBER }, line: { color: AMBER } });
  s1.addText(String(i + 1), { x: dx[i] + 0.2, y: dy + 0.22, w: 0.3, h: 0.26,
    fontFace: B, fontSize: 12, bold: true, color: INK, align: "center", margin: 0 });
  s1.addText(d[0], { x: dx[i] + 0.6, y: dy + 0.2, w: dw - 0.8, h: 0.3,
    fontFace: B, fontSize: 13, bold: true, color: WHITE, margin: 0 });
  s1.addText(d[1], { x: dx[i] + 0.2, y: dy + 0.58, w: dw - 0.4, h: 0.92,
    fontFace: B, fontSize: 9.5, color: ICE, margin: 0, lineSpacingMultiple: 1.12 });
});

// ── runtime stats
const ry = 4.78, rh = 1.05, rx = [0.55, 3.675, 6.80, 9.925], rw = 2.825;
const stats = [
  ["41.7×", "faster than realtime"],
  ["0.024", "processing ÷ video duration"],
  ["2.18 M", "runtime parameters in the head"],
  ["0", "hosted models at inference"],
];
stats.forEach((st, i) => {
  card(s1, rx[i], ry, rw, rh, PANEL2);
  s1.addText(st[0], { x: rx[i] + 0.18, y: ry + 0.12, w: rw - 0.36, h: 0.5,
    fontFace: H, fontSize: 26, bold: true, color: AMBER, margin: 0 });
  s1.addText(st[1], { x: rx[i] + 0.18, y: ry + 0.66, w: rw - 0.36, h: 0.3,
    fontFace: B, fontSize: 10.5, color: MUTE, margin: 0 });
});

s1.addText("Why this shape: Stage A alone is a complete submission. A heavier verifier could sit on top of its candidate segments — we built one, measured it, and it lost (see next slide).", {
  x: 0.55, y: 6.05, w: 12.2, h: 0.45, fontFace: B, fontSize: 11, italic: true, color: MUTE, margin: 0,
});
s1.addNotes("Cascade: frozen SigLIP encoder feeding a 2.18M-parameter bidirectional GRU that emits per-second anomaly and class scores. Segments come from hysteresis plus aggressive merging. No hosted model at runtime, as the rules require.");

// ────────────────────────────── SLIDE 2 ──────────────────────────────
const s2 = p.addSlide();
s2.background = { color: INK };

s2.addText("Result: 47.0 / 100 — and four experiments that failed", {
  x: 0.55, y: 0.34, w: 12.2, h: 0.5, fontFace: H, fontSize: 27, bold: true, color: WHITE, margin: 0,
});
s2.addText("Public arena run. Every number below is measured, not estimated.", {
  x: 0.55, y: 0.87, w: 12.2, h: 0.32, fontFace: B, fontSize: 13, color: MUTE, margin: 0,
});

// ── score tiles
const sy = 1.4, sh = 1.28, sx = [0.55, 3.68, 6.81], sw = 2.82;
const sc = [["D1 Clear event", "12.9", "/ 25"], ["D2 When it happens", "22.6", "/ 35"], ["D3 Long context", "11.5", "/ 40"]];
sc.forEach((c, i) => {
  card(s2, sx[i], sy, sw, sh, PANEL2);
  s2.addText(c[0], { x: sx[i] + 0.18, y: sy + 0.14, w: sw - 0.36, h: 0.26,
    fontFace: B, fontSize: 11, color: MUTE, margin: 0 });
  s2.addText([{ text: c[1], options: { fontSize: 30, bold: true, color: AMBER } },
              { text: "  " + c[2], options: { fontSize: 13, color: MUTE } }],
    { x: sx[i] + 0.18, y: sy + 0.5, w: sw - 0.36, h: 0.6, fontFace: H, margin: 0 });
});
card(s2, 9.94, sy, 2.81, sh, PANEL2);
s2.addText("TOTAL", { x: 10.12, y: sy + 0.14, w: 2.45, h: 0.26, fontFace: B, fontSize: 11, color: MUTE, margin: 0 });
s2.addText([{ text: "47.0", options: { fontSize: 30, bold: true, color: GREEN } },
            { text: "  / 100", options: { fontSize: 13, color: MUTE } }],
  { x: 10.12, y: sy + 0.5, w: 2.45, h: 0.6, fontFace: H, margin: 0 });

// ── what moved the score
card(s2, 0.55, 2.92, 6.1, 3.05);
s2.addText("What actually moved the score", { x: 0.75, y: 3.06, w: 5.7, h: 0.3,
  fontFace: B, fontSize: 13, bold: true, color: WHITE, margin: 0 });
const moves = [
  ["Timestamp drift bug found by inspection", "L2 matches x4", GREEN],
  ["Retune for precision, not recall", "FA 42→27", GREEN],
  ["Threshold sweep, 1 800 configs", "±0.1", MUTE],
  ["Stage B VLM — Qwen3-VL 2B and 4B zero-shot", "−3.7", RED],
  ["Longer training window (512)", "−3.5", RED],
  ["Organisers' wrong_way label corrections", "−5.2", RED],
];
moves.forEach((m, i) => {
  const y = 3.48 + i * 0.4;
  s2.addText(m[0], { x: 0.75, y, w: 4.15, h: 0.32, fontFace: B, fontSize: 10.5, color: ICE, margin: 0 });
  s2.addText(m[1], { x: 4.95, y, w: 1.52, h: 0.32, fontFace: B, fontSize: 11, bold: true,
    color: m[2], align: "right", margin: 0 });
});

// ── findings
card(s2, 6.95, 2.92, 5.8, 3.05);
s2.addText("Three things we learned", { x: 7.15, y: 3.06, w: 5.4, h: 0.3,
  fontFace: B, fontSize: 13, bold: true, color: WHITE, margin: 0 });
const finds = [
  ["A small VLM is not a free upgrade.", "Zero-shot Qwen3-VL scored 0/4 (2B) and 1/6 (4B) on segments the head got 3/6 right — at 7.4× the latency. It relabelled correct predictions to wrong ones. We kept the head."],
  ["The bottleneck is the representation, not the head.", "Two independent models — a GRU and a clip classifier at 86.8% held-out accuracy — both find exactly 9/20 on D1. Held-out train accuracy does not transfer across source domains."],
  ["We reverse-engineered the scorer.", "One submission was enough to recover the marks formula: D1 is F1-based, not the documented 0.5·binary + 0.5·class. Our local scorer now reproduces the arena exactly."],
];
finds.forEach((f, i) => {
  const y = 3.46 + i * 0.86;
  s2.addText(f[0], { x: 7.15, y, w: 5.4, h: 0.26, fontFace: B, fontSize: 11, bold: true, color: AMBER, margin: 0 });
  s2.addText(f[1], { x: 7.15, y: y + 0.26, w: 5.4, h: 0.56, fontFace: B, fontSize: 9, color: ICE,
    margin: 0, lineSpacingMultiple: 1.1 });
});

s2.addText("Assumption worth stating: train and test are separated at source-video level, so in-domain validation overstates test accuracy. That gap — not model capacity — is what caps us at 47.", {
  x: 0.55, y: 6.14, w: 12.2, h: 0.45, fontFace: B, fontSize: 11, italic: true, color: MUTE, margin: 0,
});
s2.addNotes("47.0/100 on the public arena. Biggest remaining pool is D3 at 11.5 of 40 across only four videos. Negative results are reported deliberately: the zero-shot VLM and the organisers' label correction both measurably hurt, and we kept what the measurements supported rather than what we expected.");

p.writeFile({ fileName: "deck/AHC_VAD_submission.pptx" }).then(f => console.log("wrote", f));
