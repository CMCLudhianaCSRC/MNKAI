// Build a journal-formatted .docx in the style of the International Journal of
// Ophthalmology Research (IJOR) reprint the author previously published in:
//   masthead block -> title -> authors -> structured abstract with run-in bold
//   headings -> keywords -> body with Vancouver [n] citations -> tables -> references.
//
// Reads the markdown source directly so no figure can drift in transcription.
//   node reports/manuscript/md_to_docx_ijor.js [in.md] [out.docx]
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  Header, Footer, PageNumber, convertInchesToTwip, LevelFormat,
} = require("docx");

const SRC = process.argv[2] || path.join(__dirname, "MANUSCRIPT_V2_IJOR.md");
const OUT = process.argv[3] || path.join(__dirname, "ManuscriptV2.docx");

// Journal reprints of this type are set in a serif face, small, tight leading.
const SERIF = "Cambria";
const SANS = "Calibri";
const INK = "000000";
const GREY = "595959";
const RULE = "BFBFBF";
const BODY_PT = 20;          // half-points => 10pt
const TABLE_W = 9360;        // 6.5in in DXA

const raw = fs.readFileSync(SRC, "utf8");
const lines = raw.split(/\r?\n/);

// ---------- pull the %%DIRECTIVE%% metadata block ----------
const meta = { AFFIL: [] };
const body = [];
for (const ln of lines) {
  const m = ln.match(/^%%([A-Z]+)%%(.*)$/);
  if (m) {
    if (m[1] === "AFFIL") meta.AFFIL.push(m[2].trim());
    else meta[m[1]] = m[2].trim();
  } else body.push(ln);
}

// ---------- inline markdown -> runs ----------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`\n]+`)/g;
  let last = 0, m;
  const push = (t, o) => { if (t) out.push(new TextRun({ text: t, font: SERIF, size: BODY_PT, ...base, ...o })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("`")) push(tok.slice(1, -1), { font: "Consolas", size: BODY_PT - 2 });
    else push(tok.slice(1, -1), { italics: true });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  return out.length ? out : [new TextRun({ text: "", font: SERIF, size: BODY_PT })];
}
const plain = (t) => t.replace(/\*\*/g, "").replace(/`/g, "").replace(/(^|\W)\*([^*]+)\*/g, "$1$2");

const P = (children, opts = {}) => new Paragraph({
  spacing: { before: 40, after: 100, line: 250 },
  alignment: AlignmentType.JUSTIFIED, children, ...opts,
});

const children = [];

// ================= masthead =================
children.push(new Paragraph({
  spacing: { after: 60 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: INK, space: 4 } },
  children: [new TextRun({ text: "MANUSCRIPT FOR SUBMISSION  ·  INTERIM REPORT", font: SANS, size: 15, bold: true, color: GREY, characterSpacing: 40 })],
}));
if (meta.NOTE) {
  children.push(new Paragraph({
    spacing: { before: 100, after: 160 },
    children: [new TextRun({ text: meta.NOTE, font: SANS, size: 15, italics: true, color: GREY })],
  }));
}

// ================= title / authors / affiliations =================
children.push(new Paragraph({
  spacing: { before: 120, after: 140 },
  alignment: AlignmentType.LEFT,
  children: [new TextRun({ text: meta.TITLE || "Untitled", font: SERIF, size: 32, bold: true, color: INK })],
}));
children.push(new Paragraph({
  spacing: { after: 120 },
  children: [new TextRun({ text: meta.AUTHORS || "", font: SERIF, size: 22, color: INK })],
}));
for (const a of meta.AFFIL) {
  const [who, where] = a.split("|").map(s => (s || "").trim());
  children.push(new Paragraph({
    spacing: { after: 30 },
    children: [
      new TextRun({ text: who + "  ", font: SANS, size: 15, bold: true, color: GREY }),
      new TextRun({ text: where || "", font: SANS, size: 15, color: GREY }),
    ],
  }));
}
if (meta.CORR) {
  children.push(new Paragraph({
    spacing: { before: 80, after: 140 },
    border: { top: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 4 } },
    children: [
      new TextRun({ text: "Corresponding Author: ", font: SANS, size: 15, bold: true, color: GREY }),
      new TextRun({ text: meta.CORR, font: SANS, size: 15, color: GREY }),
    ],
  }));
}

// ================= body =================
function tableAt(i) {
  if (!/^\s*\|/.test(body[i] || "")) return null;
  const rows = [];
  let j = i;
  while (j < body.length && /^\s*\|/.test(body[j])) {
    if (!/^\s*\|[\s:|-]+\|\s*$/.test(body[j])) {
      rows.push(body[j].trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim()));
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
    columnWidths: widths,
    width: { size: TABLE_W, type: WidthType.DXA },
    rows: rows.map((cells, ri) => new TableRow({
      tableHeader: ri === 0,
      children: Array.from({ length: n }, (_, ci) => new TableCell({
        width: { size: widths[ci], type: WidthType.DXA },
        shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "F2F2F2" } : undefined,
        margins: { top: 50, bottom: 50, left: 90, right: 90 },
        children: [new Paragraph({
          spacing: { before: 10, after: 10 },
          alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          children: runs(cells[ci] ?? "", { size: 18, bold: ri === 0 }),
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

let i = 0;
while (i < body.length) {
  const line = body[i];

  const tbl = tableAt(i);
  if (tbl) { children.push(buildTable(tbl.rows)); children.push(new Paragraph({ spacing: { after: 160 }, children: [] })); i = tbl.next; continue; }

  const h = line.match(/^(#{2,4})\s+(.*)$/);
  if (h) {
    const lvl = h[1].length;              // ## = section, ### = subsection
    children.push(new Paragraph({
      spacing: { before: lvl === 2 ? 260 : 190, after: 90 },
      heading: lvl === 2 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
      children: [new TextRun({
        text: plain(h[2]), font: SERIF, bold: true,
        size: lvl === 2 ? 24 : 21, color: INK, italics: lvl > 2,
      })],
    }));
    i++; continue;
  }

  // numbered reference list
  if (/^\s*\d+\.\s+/.test(line)) {
    while (i < body.length && /^\s*\d+\.\s+/.test(body[i])) {
      let t = body[i]; i++;
      while (i < body.length && body[i].trim() && !/^\s*\d+\.\s+/.test(body[i]) && !/^#{2,4}\s/.test(body[i])) { t += " " + body[i].trim(); i++; }
      children.push(new Paragraph({
        spacing: { before: 20, after: 40, line: 230 },
        indent: { left: convertInchesToTwip(0.3), hanging: convertInchesToTwip(0.3) },
        children: runs(t.trim(), { size: 18 }),
      }));
    }
    continue;
  }

  if (/^\s*[-*]\s+/.test(line)) {
    while (i < body.length && /^\s*[-*]\s+/.test(body[i])) {
      children.push(new Paragraph({
        bullet: { level: 0 }, spacing: { before: 30, after: 30, line: 240 },
        children: runs(body[i].replace(/^\s*[-*]\s+/, "")),
      }));
      i++;
    }
    continue;
  }

  if (!line.trim()) { i++; continue; }

  // paragraph — join wrapped lines
  let buf = [line]; i++;
  while (i < body.length && body[i].trim() && !/^(#{2,4}\s|\s*[-*]\s|\s*\d+\.\s|\s*\|)/.test(body[i])) { buf.push(body[i]); i++; }
  children.push(P(runs(buf.join(" ").trim())));
}

// ================= document =================
const doc = new Document({
  creator: "Ujjwal Thakran",
  title: meta.TITLE || "Manuscript V2",
  description: "Interim report, journal-formatted",
  styles: { default: { document: { run: { font: SERIF, size: BODY_PT } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },              // A4, as the journal uses
        margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 },  // 2 cm
      },
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT, spacing: { after: 100 },
        children: [new TextRun({ text: "Interim report — not for citation", font: SANS, size: 14, color: "A6A6A6" })],
      })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: SANS, size: 15, color: GREY })],
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(OUT, b);
  console.log("WROTE", OUT, `(${(b.length / 1024).toFixed(0)} KB, ${children.length} blocks)`);
});
