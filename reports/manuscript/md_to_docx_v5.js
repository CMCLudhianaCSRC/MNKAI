// Build ManuscriptV5.docx — the final research-project submission.
//
// Differences from the V4 generator:
//   * Times New Roman throughout; all headings black.
//   * References cited as SUPERSCRIPT numbers via the ^[n] marker; the numbered list
//     appears once, at the end.
//   * Figures embedded from reports/manuscript/figures via %%FIG:file|caption%%.
//   * Devanagari and Gurmukhi runs detected by Unicode range and set in Nirmala UI.
//
//   node reports/manuscript/md_to_docx_v5.js [in.md] [out.docx]
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType, HeadingLevel, PageBreak,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  Footer, PageNumber, convertInchesToTwip,
} = require("docx");

const SRC = process.argv[2] || path.join(__dirname, "MANUSCRIPT_V5_FINAL.md");
const OUT = process.argv[3] || path.join(__dirname, "ManuscriptV5.docx");
const FIGDIR = path.join(__dirname, "figures");
const PERCLASS = path.join(__dirname, "..", "..", "eval", "probe", "per_class_tables.json");

const SERIF = "Times New Roman";
const INDIC = "Nirmala UI";
const INK = "000000";
const RULE = "BFBFBF";
const BODY = 24;                 // 12pt
const TABLE_W = 9000;
const CONTENT_IN = 6.1;          // usable width in inches (A4 minus margins)

const INDIC_RE = /[ऀ-ॿ਀-੿]/;
const hasIndic = s => INDIC_RE.test(s);

function scriptRuns(text, base = {}) {
  if (!hasIndic(text)) return [new TextRun({ text, font: SERIF, size: BODY, ...base })];
  const out = []; let buf = "", cur = null;
  const flush = () => { if (buf) { out.push(new TextRun({ text: buf, font: cur ? INDIC : SERIF, size: BODY, ...base })); buf = ""; } };
  for (const ch of text) {
    const isI = INDIC_RE.test(ch);
    const cls = /\s|[.,:;()/–—-]/.test(ch) ? cur : isI;
    if (cur === null) cur = cls === null ? isI : cls;
    else if (cls !== cur) { flush(); cur = cls; }
    buf += ch;
  }
  flush();
  return out.length ? out : [new TextRun({ text, font: SERIF, size: BODY, ...base })];
}

// inline: **bold**, *italic*, and ^[1,2] -> superscript citation
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\n]+\*|\^\[[\d,\s]+\])/g;
  let last = 0, m;
  const push = (t, o) => { if (t) out.push(...scriptRuns(t, { ...base, ...o })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("^[")) {
      out.push(new TextRun({
        text: tok.slice(2, -1).replace(/\s+/g, ""),
        font: SERIF, size: BODY, superScript: true, ...base,
      }));
    } else if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else push(tok.slice(1, -1), { italics: true });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out.length ? out : [new TextRun({ text: "", font: SERIF, size: BODY })];
}
const plain = t => t.replace(/\*\*/g, "").replace(/\^\[[\d,\s]+\]/g, "").replace(/(^|\W)\*([^*]+)\*/g, "$1$2");

function perClassMarkdown(key, label) {
  if (!fs.existsSync(PERCLASS)) return `*[${label}: table pending]*`;
  const d = JSON.parse(fs.readFileSync(PERCLASS, "utf8"))[key];
  if (!d) return `*[${label}: not available]*`;
  return ["| Condition | Cases (n) | Sensitivity | Precision | F1 |",
    ...d.per_class.map(r => `| ${r.class} | ${r.support} | ${r.recall.toFixed(3)} | ${r.precision.toFixed(3)} | ${r.f1.toFixed(3)} |`)].join("\n");
}

let raw = fs.readFileSync(SRC, "utf8");
raw = raw.replace("%%TABLE_9CLASS%%", perClassMarkdown("task_9class_oof", "Nine-category model"));
raw = raw.replace("%%TABLE_25CLASS%%", perClassMarkdown("task_25class_external_clean", "Twenty-five condition model"));

const meta = {}; const L = [];
for (const ln of raw.split(/\r?\n/)) {
  const m = ln.match(/^%%([A-Z0-9_]+)%%(.*)$/);
  if (m && m[1] !== "FIG") meta[m[1]] = m[2].trim(); else L.push(ln);
}

const children = [];

// ---------------- title page ----------------
children.push(new Paragraph({
  spacing: { before: 900, after: 400 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: meta.DOCTYPE || "RESEARCH PROJECT", font: SERIF, size: 36, bold: true, color: INK, characterSpacing: 60 })],
}));
children.push(new Paragraph({
  spacing: { before: 200, after: 700 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: meta.TITLE || "", font: SERIF, size: 28, bold: true, color: INK })],
}));
function personBlock(label, val) {
  if (!val) return;
  const p = val.split("|").map(s => s.trim());
  children.push(new Paragraph({
    spacing: { before: 260, after: 40 },
    children: [
      new TextRun({ text: label, font: SERIF, size: BODY, bold: true, color: INK }),
      new TextRun({ text: "  " + (p[0] || ""), font: SERIF, size: BODY, bold: true, color: INK }),
    ],
  }));
  for (const q of p.slice(1)) children.push(new Paragraph({
    spacing: { after: 20 }, indent: { left: convertInchesToTwip(2.4) },
    children: [new TextRun({ text: q, font: SERIF, size: BODY, color: INK })],
  }));
}
personBlock("NAME OF THE CANDIDATE:", meta.CANDIDATE);
personBlock("NAME OF SUPERVISOR:", meta.SUPERVISOR);
personBlock("NAME OF CO-SUPERVISOR:", meta.COSUPERVISOR);
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------------- body ----------------
function tableAt(i) {
  if (!/^\s*\|/.test(L[i] || "")) return null;
  const rows = []; let j = i;
  while (j < L.length && /^\s*\|/.test(L[j])) {
    if (!/^\s*\|[\s:|-]+\|\s*$/.test(L[j])) rows.push(L[j].trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim()));
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
          children: runs(cells[ci] ?? "", { size: 20, bold: ri === 0 }),
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

function addFigure(file, caption) {
  const p = path.join(FIGDIR, file);
  if (!fs.existsSync(p)) {
    children.push(new Paragraph({ children: runs(`*[missing figure: ${file}]*`) }));
    return;
  }
  const dim = require("child_process");   // read PNG header for intrinsic size
  const buf = fs.readFileSync(p);
  const pw = buf.readUInt32BE(16), ph = buf.readUInt32BE(20);
  // figures were rendered at 2x for A4 content width
  const wIn = Math.min(CONTENT_IN, pw / 200);
  const hIn = wIn * (ph / pw);
  children.push(new Paragraph({
    spacing: { before: 220, after: 60 }, alignment: AlignmentType.CENTER,
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: wIn * 72, height: hIn * 72 } })],
  }));
  if (caption) children.push(new Paragraph({
    spacing: { after: 220 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: caption, font: SERIF, size: 20, italics: true, color: INK })],
  }));
}

const BREAK_BEFORE = ["ABSTRACT", "INTRODUCTION", "PRELIMINARY RESULTS", "PROFORMA",
  "APPENDIX A", "INFORMED CONSENT FORM", "सूचित सहमति", "ਸੂਚਿਤ ਸਹਿਮਤੀ",
  "PATIENT INFORMATION SHEET", "रोगी सूचना", "ਮਰੀਜ਼ ਜਾਣਕਾਰੀ", "REFERENCES"];

let i = 0;
while (i < L.length) {
  const line = L[i];

  const fig = line.match(/^%%FIG:([^|]+)\|(.*)%%$/);
  if (fig) { addFigure(fig[1].trim(), fig[2].trim()); i++; continue; }

  const t = tableAt(i);
  if (t) { children.push(buildTable(t.rows)); children.push(new Paragraph({ spacing: { after: 180 }, children: [] })); i = t.next; continue; }

  const h = line.match(/^(#{2,4})\s+(.*)$/);
  if (h) {
    const lvl = h[1].length, txt = plain(h[2]);
    if (lvl === 2 && BREAK_BEFORE.some(s => txt.startsWith(s))) children.push(new Paragraph({ children: [new PageBreak()] }));
    children.push(new Paragraph({
      spacing: { before: lvl === 2 ? 300 : 220, after: 110 },
      heading: lvl === 2 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
      children: runs(txt, { bold: true, size: lvl === 2 ? 28 : 24, color: INK }),
    }));
    i++; continue;
  }

  if (/^\s*\d+\.\s+/.test(line)) {
    while (i < L.length && /^\s*\d+\.\s+/.test(L[i])) {
      let txt = L[i]; i++;
      while (i < L.length && L[i].trim() && !/^\s*\d+\.\s+/.test(L[i]) && !/^#{2,4}\s/.test(L[i]) && !/^\s*\|/.test(L[i]) && !/^%%FIG/.test(L[i])) { txt += " " + L[i].trim(); i++; }
      children.push(new Paragraph({
        spacing: { before: 30, after: 60, line: 300 },
        indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.28) },
        children: runs(txt.trim()),
      }));
    }
    continue;
  }

  if (/^\s*[-*]\s+/.test(line)) {
    while (i < L.length && /^\s*[-*]\s+/.test(L[i])) {
      children.push(new Paragraph({
        bullet: { level: 0 }, spacing: { before: 30, after: 40, line: 300 },
        children: runs(L[i].replace(/^\s*[-*]\s+/, "")),
      }));
      i++;
    }
    continue;
  }

  if (!line.trim()) { i++; continue; }

  let buf = [line]; i++;
  while (i < L.length && L[i].trim() && !/^(#{2,4}\s|\s*[-*]\s|\s*\d+\.\s|\s*\||%%FIG)/.test(L[i])) { buf.push(L[i]); i++; }
  const text = buf.join(" ").trim();
  children.push(new Paragraph({
    spacing: { before: 60, after: 140, line: 320 },
    alignment: hasIndic(text) ? AlignmentType.LEFT : AlignmentType.JUSTIFIED,
    children: runs(text),
  }));
}

const doc = new Document({
  creator: "Ujjwal Thakran",
  title: meta.TITLE || "Research Project",
  description: "Institutional research project submission",
  styles: { default: { document: { run: { font: SERIF, size: BODY, color: INK } } } },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1701, right: 1440 } },
      titlePage: true,
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: SERIF, size: 20, color: INK })],
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
