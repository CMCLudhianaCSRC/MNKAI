// MNKAI-III — Department presentation deck generator
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
p.author = "Ujjwal Thakran";
p.title = "MNKAI-III Department Presentation";

// ---- palette (no #) ----
const DARK = "0A2F38", DARK2 = "0E3D48", TEAL = "028090", SEA = "00A896", MINT = "02C39A";
const INK = "17262B", MUTED = "5E7176", CARD = "EEF5F6", LINE = "D6E5E7", WHITE = "FFFFFF";
const ALERT = "E4572E", AMBER = "D98A00";
const HF = "Calibri", BF = "Calibri";

const shadow = () => ({ type: "outer", color: "9AB2B6", blur: 6, offset: 3, angle: 90, opacity: 0.35 });

function kicker(s, t, color = TEAL, x = 0.6, y = 0.5) {
  s.addText(t.toUpperCase(), { x, y, w: 11, h: 0.3, fontFace: HF, fontSize: 13, bold: true, color, charSpacing: 2, margin: 0 });
}
function title(s, t, x = 0.6, y = 0.82, w = 12.1, color = INK, size = 34) {
  s.addText(t, { x, y, w, h: 0.9, fontFace: HF, fontSize: size, bold: true, color, margin: 0 });
}
function card(s, x, y, w, h, fill = CARD) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.09, fill: { color: fill }, line: { color: LINE, width: 1 }, shadow: shadow() });
}
function stat(s, x, y, num, label, color = TEAL, numSize = 40, w = 3.0) {
  s.addText(num, { x, y, w, h: 0.7, fontFace: HF, fontSize: numSize, bold: true, color, align: "left", margin: 0 });
  s.addText(label, { x, y: y + 0.68, w, h: 0.6, fontFace: BF, fontSize: 12.5, color: MUTED, align: "left", margin: 0 });
}
// horizontal AUROC bar on a 0..1 scale with a chance line at 0.5
function aurocBar(s, x, y, w, value, color, label, sub) {
  const H = 0.34;
  s.addText(label, { x: x, y: y - 0.02, w: 2.6, h: H, fontFace: BF, fontSize: 13, bold: true, color: INK, align: "left", valign: "middle", margin: 0 });
  const bx = x + 2.7, bw = w;
  s.addShape(p.ShapeType.roundRect, { x: bx, y, w: bw, h: H, rectRadius: 0.04, fill: { color: "E4EDEE" }, line: { type: "none" } });
  s.addShape(p.ShapeType.roundRect, { x: bx, y, w: Math.max(0.12, bw * value), h: H, rectRadius: 0.04, fill: { color }, line: { type: "none" } });
  s.addText(value.toFixed(3), { x: bx + bw + 0.1, y, w: 1.1, h: H, fontFace: HF, fontSize: 13, bold: true, color, valign: "middle", margin: 0 });
  if (sub) s.addText(sub, { x: bx, y: y + H - 0.02, w: bw, h: 0.24, fontFace: BF, fontSize: 9.5, color: MUTED, margin: 0 });
}

// ============================================================ 1. TITLE
let s = p.addSlide(); s.background = { color: DARK };
s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: 13.33, h: 0.12, fill: { color: TEAL }, line: { type: "none" } });
s.addText("MNKAI-III", { x: 0.7, y: 1.55, w: 12, h: 1.2, fontFace: HF, fontSize: 60, bold: true, color: WHITE, margin: 0 });
s.addText("A trustworthy AI for dermatology on Indian & dark skin", { x: 0.72, y: 2.75, w: 12, h: 0.7, fontFace: HF, fontSize: 24, color: MINT, margin: 0 });
s.addText("Development & internal validation of a frozen foundation-model ensemble", { x: 0.72, y: 3.45, w: 12, h: 0.5, fontFace: BF, fontSize: 15, italic: true, color: "AEC8CC", margin: 0 });
// two-objective chips
[["OBJECTIVE 1", "Vision-model research  ·  this talk", TEAL],
 ["OBJECTIVE 2", "First North-India dataset  ·  approved (3–5k photos)", SEA]].forEach((c, i) => {
  const x = 0.72 + i * 6.0;
  s.addShape(p.ShapeType.roundRect, { x, y: 4.45, w: 5.7, h: 0.95, rectRadius: 0.1, fill: { color: DARK2 }, line: { color: TEAL, width: 1 } });
  s.addText(c[0], { x: x + 0.25, y: 4.58, w: 5.2, h: 0.3, fontFace: HF, fontSize: 12, bold: true, color: MINT, charSpacing: 2, margin: 0 });
  s.addText(c[1], { x: x + 0.25, y: 4.88, w: 5.2, h: 0.45, fontFace: BF, fontSize: 14.5, color: WHITE, margin: 0 });
});
s.addText("Ujjwal Thakran  ·  Christian Medical College, Ludhiana  ·  Department Presentation", { x: 0.72, y: 6.6, w: 12, h: 0.4, fontFace: BF, fontSize: 13, color: "8FAAAF", margin: 0 });
s.addNotes("Open: I built an AI that reads skin-lesion photos and I tested it honestly. Two objectives: (1) the AI research — today; (2) building the first North Indian skin dataset — approved for 3-5k photos. The headline: it works on dark skin, where almost every other system fails.");

// ============================================================ 2. THE PROBLEM
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "The problem");
title(s, "Dermatology AI fails on the skin it's needed for most");
card(s, 0.6, 1.95, 6.1, 4.7);
s.addText("Skin cancer and skin disease are hardest to spot on brown and dark skin — and that's exactly where AI collapses.", { x: 0.9, y: 2.25, w: 5.5, h: 1.0, fontFace: BF, fontSize: 16, color: INK, margin: 0 });
s.addText([
  { text: "Published models on the DDI dark-skin benchmark:", options: { bold: true, color: TEAL, breakLine: true, fontSize: 14, paraSpaceAfter: 6 } },
  { text: "DeepDerm — 0.50  (a coin-flip)", options: { bullet: true, color: INK, breakLine: true, fontSize: 14, paraSpaceAfter: 4 } },
  { text: "ModelDerm — 0.55", options: { bullet: true, color: INK, breakLine: true, fontSize: 14, paraSpaceAfter: 4 } },
  { text: "HAM10000 model — 0.57", options: { bullet: true, color: INK, breakLine: true, fontSize: 14 } },
], { x: 0.9, y: 3.35, w: 5.5, h: 2.0, fontFace: BF, margin: 0 });
s.addText("(AUROC — 0.5 is random guessing, 1.0 is perfect)", { x: 0.9, y: 6.05, w: 5.5, h: 0.4, fontFace: BF, fontSize: 11, italic: true, color: MUTED, margin: 0 });
// right: the data gap
card(s, 6.95, 1.95, 5.75, 4.7, DARK);
s.addText("And there is almost no data", { x: 7.25, y: 2.25, w: 5.1, h: 0.5, fontFace: HF, fontSize: 20, bold: true, color: MINT, margin: 0 });
s.addText([
  { text: "The world's skin datasets are Western — mostly light skin, mostly dermoscopy.", options: { breakLine: true, fontSize: 15, color: WHITE, paraSpaceAfter: 10 } },
  { text: "One dark-skin cancer benchmark exists (DDI) — and it's American.", options: { breakLine: true, fontSize: 15, color: WHITE, paraSpaceAfter: 10 } },
  { text: "There is NO dermatology image dataset from North India.", options: { breakLine: true, fontSize: 16, bold: true, color: AMBER } },
], { x: 7.25, y: 2.85, w: 5.15, h: 3.5, fontFace: BF, margin: 0 });
s.addNotes("Two linked problems: existing AI drops to chance on dark skin (0.50-0.57 on the DDI benchmark), and there's essentially no Indian data to fix it — none at all from North India. These two gaps define my two objectives.");

// ============================================================ 3. TWO OBJECTIVES
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "The project");
title(s, "Two objectives, one mission");
const objs = [
  ["1", "Build a trustworthy AI that works across skin tones", "Frozen foundation-model ensemble + a full safety layer, validated on diverse and dark skin. Runs on a laptop.", "DONE — this talk", TEAL],
  ["2", "Build the first North Indian dermatology dataset", "3–5k consented photos from two hospitals over a year, dermatologist-labelled, with biopsy-confirmed cancers.", "APPROVED — next phase", SEA],
];
objs.forEach((o, i) => {
  const x = 0.6 + i * 6.15;
  card(s, x, 2.0, 5.9, 4.5);
  s.addShape(p.ShapeType.ellipse, { x: x + 0.35, y: 2.35, w: 0.85, h: 0.85, fill: { color: o[4] }, line: { type: "none" } });
  s.addText(o[0], { x: x + 0.35, y: 2.37, w: 0.85, h: 0.8, fontFace: HF, fontSize: 34, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(o[1], { x: x + 1.4, y: 2.35, w: 4.3, h: 1.1, fontFace: HF, fontSize: 19, bold: true, color: INK, margin: 0 });
  s.addText(o[2], { x: x + 0.4, y: 3.65, w: 5.2, h: 1.6, fontFace: BF, fontSize: 15, color: MUTED, margin: 0 });
  s.addShape(p.ShapeType.roundRect, { x: x + 0.4, y: 5.55, w: 3.0, h: 0.55, rectRadius: 0.27, fill: { color: o[4] }, line: { type: "none" } });
  s.addText(o[3], { x: x + 0.4, y: 5.55, w: 3.0, h: 0.55, fontFace: HF, fontSize: 12.5, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
});
s.addNotes("Frame it as two objectives. Objective 1 is the AI research, done and validated internally — today's talk. Objective 2 is the dataset study, already approved — the next phase, one year.");

// ============================================================ 4. WHAT I BUILT (pipeline)
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "The system");
title(s, "One photo in, a safe clinical read out");
const steps = [
  ["Photo", "lesion image", TEAL],
  ["6 frozen AI models", "pre-trained, untouched", DARK],
  ["Light probe", "the only part we trained", SEA],
  ["Calibrated\ndifferential", "honest probabilities", TEAL],
  ["Safety gate", "confident? or refer", ALERT],
];
const bw = 2.28, gap = 0.28, y0 = 2.35;
steps.forEach((st, i) => {
  const x = 0.6 + i * (bw + gap);
  card(s, x, y0, bw, 1.5, i === 1 ? DARK : (i === 4 ? "FCEEE9" : CARD));
  const tc = i === 1 ? WHITE : INK;
  s.addText(st[0], { x: x + 0.12, y: y0 + 0.28, w: bw - 0.24, h: 0.7, fontFace: HF, fontSize: 15.5, bold: true, color: tc, align: "center", valign: "middle", margin: 0 });
  s.addText(st[1], { x: x + 0.12, y: y0 + 0.98, w: bw - 0.24, h: 0.4, fontFace: BF, fontSize: 11, color: i === 1 ? "AEC8CC" : MUTED, align: "center", margin: 0 });
  if (i < steps.length - 1) s.addShape(p.ShapeType.rightArrow, { x: x + bw + 0.01, y: y0 + 0.6, w: gap - 0.02, h: 0.3, fill: { color: SEA }, line: { type: "none" } });
});
// branch outcomes
card(s, 0.6, 4.35, 6.05, 1.7, CARD);
s.addText("Confident →", { x: 0.85, y: 4.55, w: 2.0, h: 0.4, fontFace: HF, fontSize: 15, bold: true, color: TEAL, margin: 0 });
s.addText("Decision support: top diagnoses + malignancy flag", { x: 0.85, y: 4.95, w: 5.5, h: 0.9, fontFace: BF, fontSize: 14, color: INK, margin: 0 });
card(s, 6.9, 4.35, 5.83, 1.7, "FCEEE9");
s.addText("Unsure →", { x: 7.15, y: 4.55, w: 2.0, h: 0.4, fontFace: HF, fontSize: 15, bold: true, color: ALERT, margin: 0 });
s.addText("Abstains, and the LLM asks photo-based history questions before refining", { x: 7.15, y: 4.95, w: 5.3, h: 0.9, fontFace: BF, fontSize: 14, color: INK, margin: 0 });
s.addText("Entire system runs on a 6 GB laptop — no cloud, no GPU cluster.", { x: 0.6, y: 6.35, w: 12, h: 0.4, fontFace: BF, fontSize: 14, italic: true, bold: true, color: TEAL, margin: 0 });
s.addNotes("Walk left to right: a photo goes into six frozen pre-trained models; we trained only a small 'probe' on top; that gives calibrated probabilities; then a safety gate decides. If confident, it gives decision support with a cancer flag. If unsure, it abstains and the language model asks the patient history questions. All on a laptop.");

// ============================================================ 5. CORE IDEA
s = p.addSlide(); s.background = { color: DARK };
kicker(s, "The key idea", MINT);
title(s, "Doing less beat doing more", 0.6, 0.82, 12, WHITE);
s.addText("Instead of expensively re-training a giant AI on our small data (which distorts it), we keep the pre-trained models frozen and train only a tiny classifier on top. In this data-scarce setting, the light touch wins.", { x: 0.6, y: 1.85, w: 8.0, h: 1.6, fontFace: BF, fontSize: 17, color: "D5E5E7", margin: 0 });
[["6", "frozen expert models", "each sees the lesion differently — like a tumour board", MINT],
 ["~400k", "parameters we trained", "vs 86–300 million per frozen model", SEA],
 ["minutes", "to train, on a CPU", "no GPU, no fine-tuning", TEAL]].forEach((c, i) => {
  const x = 0.6 + i * 4.15;
  s.addShape(p.ShapeType.roundRect, { x, y: 3.85, w: 3.9, h: 2.55, rectRadius: 0.1, fill: { color: DARK2 }, line: { color: TEAL, width: 1 } });
  s.addText(c[0], { x: x + 0.3, y: 4.15, w: 3.3, h: 0.9, fontFace: HF, fontSize: 40, bold: true, color: c[3], margin: 0 });
  s.addText(c[1], { x: x + 0.3, y: 5.05, w: 3.3, h: 0.5, fontFace: HF, fontSize: 15, bold: true, color: WHITE, margin: 0 });
  s.addText(c[2], { x: x + 0.3, y: 5.55, w: 3.4, h: 0.8, fontFace: BF, fontSize: 12, color: "9FBBBF", margin: 0 });
});
s.addNotes("The scientific finding: fine-tuning (retraining the whole model) HURTS when data is scarce — it overwrites the good general features. Freezing the models and training only a tiny head beats it, and runs in minutes on a laptop. Averaging six diverse frozen models is like a tumour board — the consensus beats any one.");

// ============================================================ 6. HEADLINE — dark skin cancer
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "Headline result");
title(s, "Dark-skin cancer detection at the published ceiling");
// big stat
card(s, 0.6, 2.0, 3.7, 4.5, DARK);
s.addText("0.872", { x: 0.6, y: 2.55, w: 3.7, h: 1.2, fontFace: HF, fontSize: 66, bold: true, color: MINT, align: "center", margin: 0 });
s.addText("AUROC, dark skin (FST V–VI)", { x: 0.7, y: 3.75, w: 3.5, h: 0.5, fontFace: HF, fontSize: 14, bold: true, color: WHITE, align: "center", margin: 0 });
s.addText("95% CI  0.81 – 0.93", { x: 0.7, y: 4.2, w: 3.5, h: 0.4, fontFace: BF, fontSize: 13, color: "AEC8CC", align: "center", margin: 0 });
s.addText("+0.08 vs a single model\n(p < 0.001)", { x: 0.7, y: 4.75, w: 3.5, h: 0.8, fontFace: BF, fontSize: 13, color: MINT, align: "center", margin: 0 });
// bars vs published
s.addText("Malignant-vs-benign AUROC on dark skin (DDI)", { x: 4.7, y: 2.05, w: 8, h: 0.4, fontFace: HF, fontSize: 15, bold: true, color: INK, margin: 0 });
aurocBar(s, 4.7, 2.75, 5.0, 0.50, MUTED, "DeepDerm", null);
aurocBar(s, 4.7, 3.35, 5.0, 0.55, MUTED, "ModelDerm", null);
aurocBar(s, 4.7, 3.95, 5.0, 0.57, MUTED, "HAM10000", null);
aurocBar(s, 4.7, 4.75, 5.0, 0.872, TEAL, "MNKAI-III", "6 frozen models, cross-validated on DDI");
s.addShape(p.ShapeType.line, { x: 4.7 + 2.7 + 5.0 * 0.5, y: 2.7, w: 0, h: 2.55, line: { color: ALERT, width: 1, dashType: "dash" } });
s.addText("chance = 0.5", { x: 4.7 + 2.7 + 5.0 * 0.5 - 0.6, y: 5.3, w: 1.4, h: 0.3, fontFace: BF, fontSize: 9.5, italic: true, color: ALERT, align: "center", margin: 0 });
s.addText("Honest note: published numbers are zero-shot; ours is cross-validated on DDI. Framed as reaching the achievable ceiling.", { x: 0.6, y: 6.65, w: 12.1, h: 0.4, fontFace: BF, fontSize: 10.5, italic: true, color: MUTED, margin: 0 });
s.addNotes("This is the headline. Published models sit near chance on dark skin (0.50-0.57). Ours reaches 0.872. Be honest: their numbers are zero-shot and ours is cross-validated on DDI, so I say 'we reach the achievable ceiling', not 'we beat them'. Still a large, significant, publishable result.");

// ============================================================ 7. FAIRNESS — no disparity
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "Fairness");
title(s, "The disparity disappears");
s.addText("AUROC across skin tones (our 6-model ensemble)", { x: 0.6, y: 1.95, w: 8, h: 0.4, fontFace: HF, fontSize: 15, bold: true, color: INK, margin: 0 });
aurocBar(s, 0.6, 2.6, 5.2, 0.782, SEA, "Light  (FST I–II)", "CI 0.70–0.86");
aurocBar(s, 0.6, 3.5, 5.2, 0.830, TEAL, "Mid  (FST III–IV)", "CI 0.77–0.88");
aurocBar(s, 0.6, 4.4, 5.2, 0.872, MINT, "Dark  (FST V–VI)", "CI 0.81–0.93");
s.addText("Confidence intervals overlap → no statistically significant gap by skin tone.", { x: 0.6, y: 5.5, w: 8.2, h: 0.6, fontFace: BF, fontSize: 13.5, bold: true, color: INK, margin: 0 });
// right: the gap comparison
card(s, 9.15, 2.1, 3.6, 4.3, DARK);
s.addText("Light − Dark gap", { x: 9.35, y: 2.35, w: 3.2, h: 0.4, fontFace: HF, fontSize: 15, bold: true, color: MINT, margin: 0 });
[["DeepDerm", "+0.11"], ["ModelDerm", "+0.09"], ["HAM10000", "+0.15"], ["MNKAI-III", "−0.09"]].forEach((r, i) => {
  const y = 2.9 + i * 0.82; const ours = i === 3;
  s.addText(r[0], { x: 9.35, y, w: 2.0, h: 0.5, fontFace: BF, fontSize: 14, bold: ours, color: ours ? MINT : "CFE2E4", valign: "middle", margin: 0 });
  s.addText(r[1], { x: 11.15, y, w: 1.4, h: 0.5, fontFace: HF, fontSize: 17, bold: true, color: ours ? MINT : "E79A8A", align: "right", valign: "middle", margin: 0 });
});
s.addText("Every published model is worse on dark skin. Ours is not.", { x: 9.35, y: 5.95, w: 3.2, h: 0.4, fontFace: BF, fontSize: 11, italic: true, color: "9FBBBF", margin: 0 });
s.addText("Claim carefully: “no significant disparity” — not “better on dark skin.”", { x: 0.6, y: 6.7, w: 12, h: 0.35, fontFace: BF, fontSize: 11, italic: true, color: ALERT, margin: 0 });
s.addNotes("This is the fairness story and the point of the DDI benchmark. Every published model drops on dark skin by 0.09-0.15. Ours doesn't. IMPORTANT: I say 'no significant disparity', NOT 'better on dark skin' — the intervals overlap, so claiming superiority would be overclaiming.");

// ============================================================ 8. TRUST LAYER
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "Trustworthy, not just accurate");
title(s, "It knows when it doesn't know");
const trust = [
  ["Honest confidence", "Calibration error 0.21 → 0.042", "When it says 70%, it's right ~70% of the time.", TEAL],
  ["A guaranteed short-list", "Conformal set: ~4 of 25", "Narrows 25 conditions to a small set that contains the answer 90% of the time.", SEA],
  ["Knows when to refer", "Refer 30% → 0.766 accuracy", "Hands its hardest cases to a dermatologist instead of guessing.", MINT],
  ["Flags the unfamiliar", "Out-of-distribution gate", "Rejects non-skin or unusual inputs it shouldn't judge.", DARK],
];
trust.forEach((t, i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 2.0 + Math.floor(i / 2) * 2.35;
  card(s, x, y, 5.9, 2.1);
  s.addShape(p.ShapeType.ellipse, { x: x + 0.3, y: y + 0.35, w: 0.55, h: 0.55, fill: { color: t[3] }, line: { type: "none" } });
  s.addText(t[0], { x: x + 1.05, y: y + 0.28, w: 4.6, h: 0.45, fontFace: HF, fontSize: 17, bold: true, color: INK, margin: 0 });
  s.addText(t[1], { x: x + 1.05, y: y + 0.74, w: 4.6, h: 0.4, fontFace: HF, fontSize: 14, bold: true, color: t[3] === DARK ? TEAL : t[3], margin: 0 });
  s.addText(t[2], { x: x + 0.35, y: y + 1.2, w: 5.3, h: 0.8, fontFace: BF, fontSize: 12.5, color: MUTED, margin: 0 });
});
s.addText("Most published dermatology AI reports none of these safety properties.", { x: 0.6, y: 6.75, w: 12, h: 0.35, fontFace: BF, fontSize: 12, italic: true, bold: true, color: TEAL, margin: 0 });
s.addNotes("Accuracy isn't enough for a clinical tool — it must be safe. Four safety features: honest confidence (calibration), a guaranteed short-list (conformal), knowing when to refer (abstention), and flagging weird inputs (OOD). Almost no published derm AI reports any of these — that's a real differentiator.");

// ============================================================ 9. 25 conditions
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "General classification");
title(s, "25 conditions — honest, leak-free");
stat(s, 0.7, 2.15, "0.573", "25-class macro-F1 (external, de-duplicated)\n95% CI 0.55–0.59  ·  ~14× chance", TEAL, 46, 4.0);
stat(s, 0.7, 4.15, "0.660", "9-class base task (cross-validation)\nfrozen-probe beats fine-tuning on both", SEA, 46, 4.0);
card(s, 5.3, 2.15, 7.4, 4.35, DARK);
s.addText("The honest limitation", { x: 5.6, y: 2.4, w: 6.8, h: 0.5, fontFace: HF, fontSize: 19, bold: true, color: AMBER, margin: 0 });
s.addText([
  { text: "The overall number is solid and leak-free.", options: { breakLine: true, fontSize: 15, color: WHITE, paraSpaceAfter: 10 } },
  { text: "But the dark-skin, per-condition read is underpowered — only ~135 dark-skin test images across 25 conditions (~5 each).", options: { breakLine: true, fontSize: 15, color: WHITE, paraSpaceAfter: 10 } },
  { text: "We report this as a limitation, not a claim.", options: { breakLine: true, fontSize: 15, bold: true, color: MINT, paraSpaceAfter: 10 } },
  { text: "→ This is exactly what Objective 2 (our own dataset) fixes.", options: { breakLine: true, fontSize: 15.5, bold: true, color: SEA } },
], { x: 5.6, y: 2.95, w: 6.8, h: 3.4, fontFace: BF, margin: 0 });
s.addNotes("The 25-condition task: 0.573, honest and leak-free. But be upfront — the dark-skin per-condition estimate is underpowered (only ~135 dark-skin test images). I report that as a limitation, and it's precisely the gap Objective 2 fills with our own data.");

// ============================================================ 10. HONESTY AS A FEATURE
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "Why you can trust these numbers");
title(s, "Rigour is the point, not an afterthought");
const rig = [
  ["Found & fixed 33% leakage", "Detected hidden train–test duplicates in the public benchmark and re-reported honestly (0.63 → 0.57)."],
  ["Confidence intervals on everything", "Bootstrap CIs + significance tests — not single lucky numbers."],
  ["A control experiment", "Supervised-vs-self-supervised control to bound what we can actually claim."],
  ["Deterministic, reproducible re-runs", "Seeds pinned; 89 automated tests; the whole pipeline reconstructable."],
];
rig.forEach((r, i) => {
  const y = 2.0 + i * 1.12;
  card(s, 0.6, y, 12.1, 1.0);
  s.addShape(p.ShapeType.ellipse, { x: 0.9, y: y + 0.28, w: 0.44, h: 0.44, fill: { color: TEAL }, line: { type: "none" } });
  s.addText("✓", { x: 0.9, y: y + 0.26, w: 0.44, h: 0.44, fontFace: HF, fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(r[0], { x: 1.55, y: y + 0.14, w: 4.3, h: 0.72, fontFace: HF, fontSize: 16, bold: true, color: INK, valign: "middle", margin: 0 });
  s.addText(r[1], { x: 6.0, y: y + 0.1, w: 6.5, h: 0.8, fontFace: BF, fontSize: 13, color: MUTED, valign: "middle", margin: 0 });
});
s.addNotes("This slide builds credibility. Most flashy AI papers over-hype. I caught my own 33% data-leakage bug and re-reported lower, put confidence intervals on everything, ran a control experiment, and made it reproducible with 89 tests. Honesty is the selling point to a clinical audience.");

// ============================================================ 11. DEPLOYMENT
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "Deployment");
title(s, "Practical enough for a real clinic");
[["6 GB", "laptop", "runs entirely on-device — no cloud, no GPU cluster", TEAL],
 ["4 of 6", "models to ship", "keeps 0.870 of 0.872 AUROC for ⅓ less compute", SEA],
 ["seconds", "per image", "differential + cancer flag + referral decision", MINT]].forEach((c, i) => {
  const x = 0.6 + i * 4.15;
  card(s, x, 2.0, 3.9, 2.3);
  s.addText(c[0], { x: x + 0.3, y: 2.25, w: 3.3, h: 0.85, fontFace: HF, fontSize: 38, bold: true, color: c[3], margin: 0 });
  s.addText(c[1], { x: x + 0.3, y: 3.05, w: 3.3, h: 0.4, fontFace: HF, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addText(c[2], { x: x + 0.3, y: 3.45, w: 3.4, h: 0.75, fontFace: BF, fontSize: 12, color: MUTED, margin: 0 });
});
card(s, 0.6, 4.6, 12.1, 1.95, DARK);
s.addText("The app", { x: 0.9, y: 4.8, w: 3, h: 0.5, fontFace: HF, fontSize: 18, bold: true, color: MINT, margin: 0 });
s.addText("Upload a lesion photo → ranked differential, a guaranteed short-list, a malignancy flag, and — when unsure — the AI asks history questions before it commits. Two-model switcher for base vs Indian tasks.", { x: 0.9, y: 5.3, w: 11.3, h: 1.1, fontFace: BF, fontSize: 15, color: WHITE, margin: 0 });
s.addNotes("It's not just a lab result — it's deployable. Runs on a 6GB laptop, you can ship 4 of the 6 models for a third less compute with basically no accuracy loss, and returns a full read in seconds. Then I'll show the live app.");

// ============================================================ 12. LIVE DEMO (section)
s = p.addSlide(); s.background = { color: DARK };
s.addShape(p.ShapeType.ellipse, { x: 5.5, y: 2.0, w: 2.3, h: 2.3, fill: { color: TEAL }, line: { type: "none" }, shadow: shadow() });
s.addShape(p.ShapeType.triangle, { x: 6.4, y: 2.75, w: 0.85, h: 0.8, fill: { color: WHITE }, line: { type: "none" }, rotate: 90 });
s.addText("Live demo", { x: 0.6, y: 4.5, w: 12.1, h: 0.9, fontFace: HF, fontSize: 44, bold: true, color: WHITE, align: "center", margin: 0 });
s.addText("localhost:8800  —  upload a lesion, watch it produce a differential, flag malignancy, and abstain when unsure", { x: 1.0, y: 5.5, w: 11.3, h: 0.6, fontFace: BF, fontSize: 15, color: "AEC8CC", align: "center", margin: 0 });
s.addNotes("DEMO SCRIPT: (1) Open localhost:8800. (2) Upload a clear benign lesion — show the confident differential + low malignancy. (3) Upload a suspicious/ambiguous one — show it flags malignancy and/or abstains and asks history questions. (4) Point out the conformal short-list and the referral recommendation. Keep it to ~3 minutes.");

// ============================================================ 13. OBJECTIVE 2 — dataset
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "Objective 2  ·  approved", SEA);
title(s, "The first North Indian dermatology dataset");
const d2 = [
  ["Why", "No North Indian skin dataset exists. DDI is American; DermaCon-IN is South India with no cancer arm.", TEAL],
  ["What", "3–5k consented photos · 2 hospitals · 1 year · dermatologist labels · biopsy-confirmed cancers.", SEA],
  ["How", "Standardised smartphone capture + a simple app + a clip-on dermoscope. No costly infrastructure.", MINT],
  ["Proven path", "DermaCon-IN (S. India) & PAD-UFES-20 (Brazil) did exactly this. Ours adds North India + a cancer arm.", DARK],
];
d2.forEach((c, i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 2.0 + Math.floor(i / 2) * 2.3;
  card(s, x, y, 5.9, 2.05);
  s.addText(c[0], { x: x + 0.35, y: y + 0.25, w: 5.2, h: 0.5, fontFace: HF, fontSize: 17, bold: true, color: c[2] === DARK ? TEAL : c[2], margin: 0 });
  s.addText(c[1], { x: x + 0.35, y: y + 0.8, w: 5.3, h: 1.1, fontFace: BF, fontSize: 14, color: INK, margin: 0 });
});
s.addText("Publishable as a dataset paper on its own — and it turns “works on diverse skin” into “works on Indian patients.”", { x: 0.6, y: 6.7, w: 12, h: 0.4, fontFace: BF, fontSize: 12, italic: true, bold: true, color: SEA, margin: 0 });
s.addNotes("Objective 2, already approved. Why: no North Indian data exists. What: 3-5k photos, two hospitals, a year, with biopsy-confirmed cancers. How: smartphones + app + dermoscope. It follows a proven path (South India's DermaCon-IN, Brazil's PAD-UFES-20) but adds North India and a cancer arm nobody has.");

// ============================================================ 14. ROADMAP
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "Timeline");
title(s, "Where this goes");
const phases = [
  ["Phase 1", "The AI — done", "Frozen-model ensemble, trust layer, internal validation on diverse & dark skin.", TEAL, "NOW"],
  ["Phase 2", "The dataset — 1 year", "Collect 3–5k North Indian photos with biopsy-confirmed cancers.", SEA, "APPROVED"],
  ["Outputs", "Two papers", "(1) the AI method + fairness result  ·  (2) the North Indian dataset.", DARK, "GOAL"],
];
phases.forEach((ph, i) => {
  const x = 0.6 + i * 4.15;
  card(s, x, 2.2, 3.9, 3.6);
  s.addShape(p.ShapeType.roundRect, { x: x + 0.3, y: 2.5, w: 1.6, h: 0.5, rectRadius: 0.24, fill: { color: ph[3] }, line: { type: "none" } });
  s.addText(ph[4], { x: x + 0.3, y: 2.5, w: 1.6, h: 0.5, fontFace: HF, fontSize: 11, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(ph[0], { x: x + 0.3, y: 3.15, w: 3.3, h: 0.5, fontFace: HF, fontSize: 22, bold: true, color: INK, margin: 0 });
  s.addText(ph[1], { x: x + 0.3, y: 3.7, w: 3.3, h: 0.45, fontFace: HF, fontSize: 15, bold: true, color: ph[3] === DARK ? TEAL : ph[3], margin: 0 });
  s.addText(ph[2], { x: x + 0.3, y: 4.25, w: 3.4, h: 1.4, fontFace: BF, fontSize: 13, color: MUTED, margin: 0 });
  if (i < 2) s.addShape(p.ShapeType.rightArrow, { x: x + 3.92, y: 3.75, w: 0.22, h: 0.35, fill: { color: SEA }, line: { type: "none" } });
});
s.addText("Publishing Phase 1 first de-risks Phase 2: a proven method + a clear data gap is the strongest possible case for the dataset.", { x: 0.6, y: 6.15, w: 12.1, h: 0.6, fontFace: BF, fontSize: 13, italic: true, color: TEAL, margin: 0 });
s.addNotes("Phase 1 is done — today. Phase 2 is the approved one-year dataset study. Together they produce two papers. Key line for the committee: publishing Phase 1 first actually strengthens the Phase 2 proposal — a working method plus a clear gap is far more fundable than an idea.");

// ============================================================ 15. LIMITATIONS
s = p.addSlide(); s.background = { color: WHITE };
kicker(s, "Honest limitations", AMBER);
title(s, "What this is — and isn't — yet");
const lims = [
  ["Decision support, not diagnosis", "A safety-first assistant that flags and refers; the dermatologist always decides."],
  ["Internal validation only", "Cross-validated, not yet prospectively validated on new patients — that's Phase 2."],
  ["Dark-skin cancer set is small & American", "DDI: 656 images, one US clinic. Strong but data-limited; wide intervals."],
  ["25-class dark-skin is underpowered", "~135 dark-skin test images — reported as a limitation, fixed by our dataset."],
];
lims.forEach((l, i) => {
  const y = 2.0 + i * 1.12;
  card(s, 0.6, y, 12.1, 1.0, "FBF3E6");
  s.addShape(p.ShapeType.ellipse, { x: 0.9, y: y + 0.28, w: 0.44, h: 0.44, fill: { color: AMBER }, line: { type: "none" } });
  s.addText("!", { x: 0.9, y: y + 0.26, w: 0.44, h: 0.44, fontFace: HF, fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
  s.addText(l[0], { x: 1.55, y: y + 0.13, w: 5.2, h: 0.74, fontFace: HF, fontSize: 15.5, bold: true, color: INK, valign: "middle", margin: 0 });
  s.addText(l[1], { x: 6.9, y: y + 0.1, w: 5.6, h: 0.8, fontFace: BF, fontSize: 12.5, color: "6B5A3A", valign: "middle", margin: 0 });
});
s.addNotes("Being upfront about limits builds trust with clinicians. It's decision support, not diagnosis. It's internally validated, not prospectively. The dark-skin cancer set is small and American. The 25-class dark-skin read is underpowered. Every one of these points straight to why Phase 2 matters.");

// ============================================================ 16. SUMMARY / THANKS
s = p.addSlide(); s.background = { color: DARK };
s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: 13.33, h: 0.12, fill: { color: TEAL }, line: { type: "none" } });
kicker(s, "In one line", MINT);
s.addText("Trustworthy dermatology AI for dark skin — on a laptop.", { x: 0.6, y: 0.95, w: 12.1, h: 1.4, fontFace: HF, fontSize: 34, bold: true, color: WHITE, margin: 0 });
[["0.872", "dark-skin cancer AUROC", MINT],
 ["No", "significant disparity by skin tone", SEA],
 ["Full", "calibration + referral safety layer", TEAL],
 ["First", "North India dataset — next", AMBER]].forEach((c, i) => {
  const x = 0.6 + i * 3.05;
  s.addShape(p.ShapeType.roundRect, { x, y: 2.7, w: 2.85, h: 1.9, rectRadius: 0.1, fill: { color: DARK2 }, line: { color: TEAL, width: 1 } });
  s.addText(c[0], { x: x + 0.2, y: 2.95, w: 2.45, h: 0.85, fontFace: HF, fontSize: 30, bold: true, color: c[2], align: "center", margin: 0 });
  s.addText(c[1], { x: x + 0.15, y: 3.85, w: 2.55, h: 0.7, fontFace: BF, fontSize: 12.5, color: WHITE, align: "center", margin: 0 });
});
s.addText("Thank you", { x: 0.6, y: 5.15, w: 12.1, h: 0.8, fontFace: HF, fontSize: 30, bold: true, color: WHITE, align: "center", margin: 0 });
s.addText("Ujjwal Thakran  ·  CMC Ludhiana  ·  Questions welcome", { x: 0.6, y: 5.95, w: 12.1, h: 0.5, fontFace: BF, fontSize: 14, color: "9FBBBF", align: "center", margin: 0 });
s.addNotes("Close on the one-liner: trustworthy dermatology AI for dark skin, on a laptop. Four takeaways: 0.872 on dark-skin cancer, no disparity, a full safety layer, and the North India dataset next. Invite questions.");

p.writeFile({ fileName: "reports/presentation/MNKAI-III_Department.pptx" }).then(f => console.log("WROTE", f));
