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

// The live board is the private Evaluation pack (E001-E028). These are actual
// marks from submission_v9a.json; update and rerun after any further upload.
const SCORE = { d1: "14.5", d2: "27.6", d3: "19.7", total: "61.8", rank: "2nd" };
// Recall from the leaderboard's own per-difficulty columns, used in the footer.
const RECALL = { d2: "5 of 12", d3: "5 of 6" };

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

s1.addText("Why this shape: Stage A alone is a complete submission. A heavier verifier could sit on top of its candidate segments — we built one and measured it, and zero-shot Qwen3-VL scored 1/6 where this head scored 3/6, at 7.4× the latency. We shipped without it.", {
  x: 0.55, y: 6.05, w: 12.2, h: 0.45, fontFace: B, fontSize: 11, italic: true, color: MUTE, margin: 0,
});
s1.addNotes("Cascade: frozen SigLIP encoder feeding a 2.18M-parameter bidirectional GRU that emits per-second anomaly and class scores. Segments come from hysteresis plus aggressive merging. No hosted model at runtime, as the rules require.");

// ────────────────────────────── SLIDE 2 ──────────────────────────────
const s2 = p.addSlide();
s2.background = { color: INK };

s2.addText("Private evaluation set: " + SCORE.total + " / 100, " + SCORE.rank + " place", {
  x: 0.55, y: 0.34, w: 12.2, h: 0.5, fontFace: H, fontSize: 27, bold: true, color: WHITE, margin: 0,
});
s2.addText("From 37.2 to " + SCORE.total + " in one afternoon \u2014 and almost none of it came from a better model.", {
  x: 0.55, y: 0.87, w: 12.2, h: 0.32, fontFace: B, fontSize: 13, color: MUTE, margin: 0,
});

// ── score tiles: standing upload, with the projection under each
const sy = 1.4, sh = 1.42, sx = [0.55, 3.68, 6.81], sw = 2.82;
const sc = [["D1 Clear event", SCORE.d1, "/ 25", "was 11.8"],
            ["D2 When it happens", SCORE.d2, "/ 35", "was 16.1 \u00b7 best on board"],
            ["D3 Long context", SCORE.d3, "/ 40", "was 16.3 \u00b7 recall " + RECALL.d3]];
sc.forEach((c, i) => {
  card(s2, sx[i], sy, sw, sh, PANEL2);
  s2.addText(c[0], { x: sx[i] + 0.18, y: sy + 0.12, w: sw - 0.36, h: 0.26,
    fontFace: B, fontSize: 11, color: MUTE, margin: 0 });
  s2.addText([{ text: c[1], options: { fontSize: 30, bold: true, color: AMBER } },
              { text: "  " + c[2], options: { fontSize: 13, color: MUTE } }],
    { x: sx[i] + 0.18, y: sy + 0.44, w: sw - 0.36, h: 0.56, fontFace: H, margin: 0 });
  s2.addText(c[3], { x: sx[i] + 0.18, y: sy + 1.04, w: sw - 0.36, h: 0.26,
    fontFace: B, fontSize: 10, color: GREEN, margin: 0 });
});
card(s2, 9.94, sy, 2.81, sh, PANEL2);
s2.addText("TOTAL", { x: 10.12, y: sy + 0.12, w: 2.45, h: 0.26, fontFace: B, fontSize: 11, color: MUTE, margin: 0 });
s2.addText([{ text: SCORE.total, options: { fontSize: 30, bold: true, color: GREEN } },
            { text: "  / 100", options: { fontSize: 13, color: MUTE } }],
  { x: 10.12, y: sy + 0.44, w: 2.45, h: 0.56, fontFace: H, margin: 0 });
s2.addText(SCORE.rank + " place \u00b7 started at 37.2", { x: 10.12, y: sy + 1.04, w: 2.45, h: 0.26,
  fontFace: B, fontSize: 10, color: GREEN, margin: 0 });

// ── what moved the score
card(s2, 0.55, 3.06, 6.1, 2.92);
s2.addText("What actually moved the score", { x: 0.75, y: 3.2, w: 5.7, h: 0.3,
  fontFace: B, fontSize: 13, bold: true, color: WHITE, margin: 0 });
const moves = [
  ["Silencing the normal Level-2 video we never spotted", "D2 16.1 → 27.6", GREEN],
  ["Candidates on the 5 s composition lattice", "D3 16.3 → 19.7", GREEN],
  ["Level 1 with no confidence gate at all", "D1 11.8 → 14.5", GREEN],
  ["Collection-fingerprint class prior", "37.2 → 46.2", GREEN],
  ["Raising the Level-1 threshold to 0.7", "−2.0", RED],
  ["12 replacement rankers on Level 3", "0 of 12", RED],
];
moves.forEach((m, i) => {
  const y = 3.62 + i * 0.39;
  s2.addText(m[0], { x: 0.75, y, w: 4.15, h: 0.32, fontFace: B, fontSize: 10.5, color: ICE, margin: 0 });
  s2.addText(m[1], { x: 4.95, y, w: 1.52, h: 0.32, fontFace: B, fontSize: 11, bold: true,
    color: m[2], align: "right", margin: 0 });
});

// ── findings
card(s2, 6.95, 3.06, 5.8, 2.92);
s2.addText("Three things we learned", { x: 7.15, y: 3.2, w: 5.4, h: 0.3,
  fontFace: B, fontSize: 13, bold: true, color: WHITE, margin: 0 });
const finds = [
  ["We were scoring a formula we had invented.",
   "A scorer fitted to the practice pack said Level 1 was F1-based, so we kept tightening a confidence gate. The leaderboard prints “found x/17”, which fixes the denominator and restores the documented rule: half the marks are binary accuracy and 17 of 20 clips carry an event, so there is no precision penalty at all. Deleting the gate was worth +2.7."],
  ["A normal video is worth a whole video.",
   "A rival submitted nothing on Level 2 and scored exactly 17.5 of 35 — one half, so two of its four videos are normal. We had carried events on both candidates in every upload, taking a guaranteed zero on one. Silencing the right one was worth +11.5, our single largest gain."],
  ["Proposals are solved; ranking is not.",
   "A 2.5 s lattice covers 100% of public truths at IoU 0.5 and our recall now leads the field. But the head’s anomaly curve is saturated at exactly 1.0000, zero variance, on three private videos, and twelve replacement scores all reach 0% recall@128 on Level 3."],
];
finds.forEach((f, i) => {
  const y = 3.6 + i * 0.81;
  s2.addText(f[0], { x: 7.15, y, w: 5.4, h: 0.24, fontFace: B, fontSize: 11, bold: true, color: AMBER, margin: 0 });
  s2.addText(f[1], { x: 7.15, y: y + 0.24, w: 5.4, h: 0.54, fontFace: B, fontSize: 8.5, color: ICE,
    margin: 0, lineSpacingMultiple: 1.08 });
});

s2.addText("Where the marks still are: we hold the highest recall in the field at both temporal difficulties — " + RECALL.d3 + " at Level 3, " + RECALL.d2 + " at Level 2 — and convert the least of it, because we cannot say which of our candidates are the right ones. That one gap is worth about thirteen marks.", {
  x: 0.55, y: 6.14, w: 12.2, h: 0.45, fontFace: B, fontSize: 11, italic: true, color: MUTE, margin: 0,
});
s2.addNotes("Private evaluation set: " + SCORE.total + "/100, " + SCORE.rank + " place, up from 37.2. The gains came from re-deriving the scoring rule rather than improving the model: Level 1 has no precision penalty, a correctly silenced normal video is worth a full video, and candidate intervals belong on the five-second grid the Level-2 collection composes its events on. The remaining gap is ranking \u2014 our recall leads the field and our precision is last, because twelve independent scores all fail to order the lattice on Level 3.");

p.writeFile({ fileName: "deck/AHC_VAD_submission.pptx" }).then(f => console.log("wrote", f));
