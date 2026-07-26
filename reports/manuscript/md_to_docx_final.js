// Build the FINAL institutional research-project submission (ManuscriptV4.docx),
// covering every item of the departmental Checklist for Assessment of Research Proposal.
//
// Differs from the V3 generator in two respects:
//   1. Devanagari (Hindi) and Gurmukhi (Punjabi) runs are detected by Unicode range and
//      set in Nirmala UI, which ships with Windows and renders both scripts correctly.
//      Latin text inside those paragraphs keeps the serif face.
//   2. Every annexure (consent forms in three languages, patient information sheets in
//      three languages, references) starts on a fresh page.
//
//   node reports/manuscript/md_to_docx_final.js [in.md] [out.docx]
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel, PageBreak,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  Footer, PageNumber, convertInchesToTwip,
} = require("docx");

const SRC = process.argv[2] || path.join(__dirname, "MANUSCRIPT_V4_FINAL.md");
const OUT = process.argv[3] || path.join(__dirname, "ManuscriptV4.docx");
const PERCLASS = path.join(__dirname, "..", "..", "eval", "probe", "per_class_tables.json");

const SERIF = "Cambria";
const INDIC = "Nirmala UI";       // renders both Devanagari and Gurmukhi on Windows
const INK = "000000";
const GREY = "595959";
const RULE = "BFBFBF";
const BODY = 22;
const TABLE_W = 9000;

// Devanagari U+0900–U+097F, Gurmukhi U+0A00–U+0A7F
const INDIC_RE = /[ऀ-ॿ਀-੿]/;
const hasIndic = s => INDIC_RE.test(s);

// split a string into runs so Indic segments get Nirmala UI and Latin keeps the serif
function scriptRuns(text, base = {}) {
  if (!hasIndic(text)) return [new TextRun({ text, font: SERIF, size: BODY, ...base })];
  const out = [];
  let buf = "", bufIndic = null;
  const flush = () => {
    if (!buf) return;
    out.push(new TextRun({ text: buf, font: bufIndic ? INDIC : SERIF, size: BODY, ...base }));
    buf = "";
  };
  for (const ch of text) {
    const isI = INDIC_RE.test(ch);
    // punctuation/space inherits the current run so we don't fragment needlessly
    const cls = /\s|[.,:;()/–—-]/.test(ch) ? bufIndic : isI;
    if (bufIndic === null) bufIndic = cls === null ? isI : cls;
    else if (cls !== bufIndic) { flush(); bufIndic = cls; }
    buf += ch;
  }
  flush();
  return out.length ? out : [new TextRun({ text, font: SERIF, size: BODY, ...base })];
}

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\n]+\*)/g;
  let last = 0, m;
  const push = (t, o) => { if (t) out.push(...scriptRuns(t, { ...base, ...o })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else push(tok.slice(1, -1), { italics: true });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out.length ? out : [new TextRun({ text: "", font: SERIF, size: BODY })];
}
const plain = t => t.replace(/\*\*/g, "").replace(/(^|\W)\*([^*]+)\*/g, "$1$2");

// ---------------- inject per-class tables ----------------
function perClassMarkdown(key, label) {
  if (!fs.existsSync(PERCLASS)) return `*[${label}: per-class table pending]*`;
  const d = JSON.parse(fs.readFileSync(PERCLASS, "utf8"))[key];
  if (!d) return `*[${label}: not available]*`;
  return ["| Condition | Cases (n) | Sensitivity | Precision | F1 |",
    ...d.per_class.map(r => `| ${r.class} | ${r.support} | ${r.recall.toFixed(3)} | ${r.precision.toFixed(3)} | ${r.f1.toFixed(3)} |`)].join("\n");
}

let raw = fs.readFileSync(SRC, "utf8");
raw = raw.replace("%%TABLE_9CLASS%%", perClassMarkdown("task_9class_oof", "Nine-category model"));
raw = raw.replace("%%TABLE_25CLASS%%", perClassMarkdown("task_25class_external_clean", "Twenty-five condition model"));

const meta = {}; const bodyLines = [];
for (const ln of raw.split(/\r?\n/)) {
  const m = ln.match(/^%%([A-Z0-9_]+)%%(.*)$/);
  if (m) meta[m[1]] = m[2].trim(); else bodyLines.push(ln);
}

const children = [];

// ================= TITLE PAGE =================
children.push(new Paragraph({
  spacing: { before: 900, after: 400 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: meta.DOCTYPE || "RESEARCH PROJECT", font: SERIF, size: 36, bold: true, characterSpacing: 60 })],
}));
children.push(new Paragraph({
  spacing: { before: 200, after: 700 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: meta.TITLE || "", font: SERIF, size: 26, bold: true })],
}));
function personBlock(label, val) {
  if (!val) return;
  const parts = val.split("|").map(s => s.trim());
  children.push(new Paragraph({
    spacing: { before: 260, after: 40 },
    children: [
      new TextRun({ text: label, font: SERIF, size: BODY, bold: true }),
      new TextRun({ text: "  " + (parts[0] || ""), font: SERIF, size: BODY, bold: true }),
    ],
  }));
  for (const p of parts.slice(1)) {
    children.push(new Paragraph({
      spacing: { after: 20 }, indent: { left: convertInchesToTwip(2.4) },
      children: [new TextRun({ text: p, font: SERIF, size: BODY })],
    }));
  }
}
personBlock("NAME OF THE CANDIDATE:", meta.CANDIDATE);
personBlock("NAME OF SUPERVISOR:", meta.SUPERVISOR);
personBlock("NAME OF CO-SUPERVISOR:", meta.COSUPERVISOR);
children.push(new Paragraph({ children: [new PageBreak()] }));

// ================= BODY =================
function tableAt(i) {
  if (!/^\s*\|/.test(bodyLines[i] || "")) return null;
  const rows = []; let j = i;
  while (j < bodyLines.length && /^\s*\|/.test(bodyLines[j])) {
    if (!/^\s*\|[\s:|-]+\|\s*$/.test(bodyLines[j])) {
      rows.push(bodyLines[j].trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim()));
    }
    j++;
  }
  return rows.length ? { rows, next: j } : null;
}

function buildTable(rows) {
  const n = Math.max(...rows.map(r => r.length));
  const w = Math.floor(TABLE_W / n);
  const widths = Array(n).fill(w); widths[n - 1] = TABLE_W - w * (n - 1);
  return new Table({
    columnWidths: widths, width: { size: TABLE_W, type: WidthType.DXA },
    rows: rows.map((cells, ri) => new TableRow({
      tableHeader: ri === 0,
      children: Array.from({ length: n }, (_, ci) => new TableCell({
        width: { size: widths[ci], type: WidthType.DXA },
        shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "EFEFEF" } : undefined,
        margins: { top: 50, bottom: 50, left: 90, right: 90 },
        children: [new Paragraph({
          spacing: { before: 10, after: 10 },
          alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          children: runs(cells[ci] ?? "", { size: 19, bold: ri === 0 }),
        })],
      })),
    })),
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: INK },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: INK },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: RULE },
      insideVertical: { style: BorderStyle.NONE },
    },
  });
}

// annexures and each translated form begin on a fresh page
const BREAK_BEFORE = ["ABSTRACT", "INTRODUCTION", "PRELIMINARY RESULTS", "PROFORMA",
  "INFORMED CONSENT FORM", "सूचित सहमति", "ਸੂਚਿਤ ਸਹਿਮਤੀ",
  "PATIENT INFORMATION SHEET", "रोगी सूचना", "ਮਰੀਜ਼ ਜਾਣਕਾਰੀ", "REFERENCES"];

let i = 0;
while (i < bodyLines.length) {
  const line = bodyLines[i];

  const t = tableAt(i);
  if (t) { children.push(buildTable(t.rows)); children.push(new Paragraph({ spacing: { after: 180 }, children: [] })); i = t.next; continue; }

  const h = line.match(/^(#{2,4})\s+(.*)$/);
  if (h) {
    const lvl = h[1].length, txt = plain(h[2]);
    if (lvl === 2 && BREAK_BEFORE.some(s => txt.startsWith(s))) {
      children.push(new Paragraph({ children: [new PageBreak()] }));
    }
    children.push(new Paragraph({
      spacing: { before: lvl === 2 ? 300 : 220, after: 110 },
      heading: lvl === 2 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
      children: runs(txt, { bold: true, size: lvl === 2 ? 26 : 22 }),
    }));
    i++; continue;
  }

  if (/^\s*\d+\.\s+/.test(line)) {
    while (i < bodyLines.length && /^\s*\d+\.\s+/.test(bodyLines[i])) {
      let txt = bodyLines[i]; i++;
      while (i < bodyLines.length && bodyLines[i].trim() && !/^\s*\d+\.\s+/.test(bodyLines[i]) && !/^#{2,4}\s/.test(bodyLines[i]) && !/^\s*\|/.test(bodyLines[i])) { txt += " " + bodyLines[i].trim(); i++; }
      children.push(new Paragraph({
        spacing: { before: 30, after: 60, line: 280 },
        indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.28) },
        children: runs(txt.trim()),
      }));
    }
    continue;
  }

  if (/^\s*[-*]\s+/.test(line)) {
    while (i < bodyLines.length && /^\s*[-*]\s+/.test(bodyLines[i])) {
      children.push(new Paragraph({
        bullet: { level: 0 }, spacing: { before: 30, after: 40, line: 280 },
        children: runs(bodyLines[i].replace(/^\s*[-*]\s+/, "")),
      }));
      i++;
    }
    continue;
  }

  if (!line.trim()) { i++; continue; }

  let buf = [line]; i++;
  while (i < bodyLines.length && bodyLines[i].trim() && !/^(#{2,4}\s|\s*[-*]\s|\s*\d+\.\s|\s*\|)/.test(bodyLines[i])) { buf.push(bodyLines[i]); i++; }
  const text = buf.join(" ").trim();
  children.push(new Paragraph({
    spacing: { before: 60, after: 130, line: 300 },
    // Harvard reference list is hanging-indented; Indic paragraphs are left-aligned
    alignment: hasIndic(text) ? AlignmentType.LEFT : AlignmentType.JUSTIFIED,
    children: runs(text),
  }));
}

const doc = new Document({
  creator: "Ujjwal Thakran",
  title: meta.TITLE || "Research Project",
  description: "Institutional research project submission — checklist-complete",
  styles: { default: { document: { run: { font: SERIF, size: BODY } } } },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1701, right: 1440 } },
      titlePage: true,
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: SERIF, size: 20, color: GREY })],
      })] }),
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(OUT, b);
  console.log("WROTE", OUT, `(${(b.length / 1024).toFixed(0)} KB, ${children.length} blocks)`);
});
