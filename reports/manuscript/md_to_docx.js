// Convert reports/manuscript/MANUSCRIPT.md -> a properly formatted Word .docx
// Reads the markdown file directly so no number can drift in transcription.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, PageBreak,
  Header, Footer, PageNumber, LevelFormat, convertInchesToTwip,
} = require("docx");

const SRC = process.argv[2] || path.join(__dirname, "MANUSCRIPT.md");
const OUT = process.argv[3] || path.join(__dirname, "MNKAI-III_Manuscript.docx");

const BODY = "Cambria";      // serif body — academic
const SANS = "Calibri";      // headings/labels
const ACCENT = "0E3D48";     // deep teal, matches the project identity
const RULE = "C9D8DA";

const TABLE_W = 9360;        // 6.5" content width in DXA

// ---------- inline markdown (**bold**, *italic*, `code`) -> TextRuns ----------
function runs(text, base = {}) {
  const out = [];
  // tokenizer: **bold**, *italic* or _italic_, `code`
  const re = /(\*\*[^*]+\*\*|\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`)/g;
  let last = 0, m;
  const push = (t, opts) => { if (t) out.push(new TextRun({ text: t, font: BODY, size: 22, ...base, ...opts })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("`")) push(tok.slice(1, -1), { font: "Consolas", size: 20 });
    else push(tok.slice(1, -1), { italics: true });
    last = m.index + tok.length;
  }
  push(text.slice(last));
  if (out.length === 0) push("");
  return out;
}

function stripInline(t) {
  return t.replace(/\*\*/g, "").replace(/`/g, "").replace(/(^|\W)\*([^*]+)\*/g, "$1$2");
}

// ---------- parse markdown ----------
const lines = fs.readFileSync(SRC, "utf8").split(/\r?\n/);
const children = [];
let i = 0;

function tableAt(start) {                       // returns {rows, next} or null
  if (!/^\s*\|/.test(lines[start] || "")) return null;
  if (!/^\s*\|[\s:|-]+\|\s*$/.test(lines[start + 1] || "")) return null;
  const rows = [];
  let j = start;
  while (j < lines.length && /^\s*\|/.test(lines[j])) {
    if (!/^\s*\|[\s:|-]+\|\s*$/.test(lines[j])) {
      rows.push(lines[j].trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim()));
    }
    j++;
  }
  return { rows, next: j };
}

function buildTable(rows) {
  const nCols = Math.max(...rows.map(r => r.length));
  const colW = Math.floor(TABLE_W / nCols);
  const widths = Array(nCols).fill(colW);
  widths[nCols - 1] = TABLE_W - colW * (nCols - 1);   // make them sum exactly

  const trs = rows.map((cells, ri) => new TableRow({
    tableHeader: ri === 0,
    children: Array.from({ length: nCols }, (_, ci) => {
      const raw = cells[ci] ?? "";
      return new TableCell({
        width: { size: widths[ci], type: WidthType.DXA },
        shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "EDF3F4" } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({
          spacing: { before: 20, after: 20 },
          children: runs(raw, ri === 0 ? { bold: true, font: SANS, size: 19 } : { size: 19 }),
        })],
      });
    }),
  }));

  return new Table({
    columnWidths: widths,
    width: { size: TABLE_W, type: WidthType.DXA },
    rows: trs,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      right: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: RULE },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: RULE },
    },
  });
}

while (i < lines.length) {
  const line = lines[i];

  // table
  const tbl = tableAt(i);
  if (tbl) {
    children.push(buildTable(tbl.rows));
    children.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
    i = tbl.next;
    continue;
  }

  // horizontal rule -> thin spacer rule
  if (/^\s*---+\s*$/.test(line)) {
    children.push(new Paragraph({
      spacing: { before: 120, after: 200 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 1 } },
      children: [],
    }));
    i++; continue;
  }

  // headings
  const h = line.match(/^(#{1,4})\s+(.*)$/);
  if (h) {
    const lvl = h[1].length;
    const txt = stripInline(h[2]);
    if (lvl === 1) {
      children.push(new Paragraph({
        spacing: { before: 0, after: 240 },
        children: [new TextRun({ text: txt, font: SANS, size: 40, bold: true, color: ACCENT })],
      }));
    } else {
      children.push(new Paragraph({
        heading: lvl === 2 ? HeadingLevel.HEADING_1 : lvl === 3 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
        spacing: { before: lvl === 2 ? 320 : 240, after: 120 },
        children: [new TextRun({
          text: txt, font: SANS, bold: true,
          size: lvl === 2 ? 30 : lvl === 3 ? 25 : 23,
          color: lvl === 2 ? ACCENT : "17262B",
        })],
      }));
    }
    i++; continue;
  }

  // bullet list
  if (/^\s*[-*]\s+/.test(line)) {
    while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
      children.push(new Paragraph({
        bullet: { level: 0 },
        spacing: { before: 40, after: 40, line: 280 },
        children: runs(lines[i].replace(/^\s*[-*]\s+/, "")),
      }));
      i++;
    }
    continue;
  }

  // numbered list
  if (/^\s*\d+\.\s+/.test(line)) {
    while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
      children.push(new Paragraph({
        numbering: { reference: "num", level: 0 },
        spacing: { before: 40, after: 40, line: 280 },
        children: runs(lines[i].replace(/^\s*\d+\.\s+/, "")),
      }));
      i++;
    }
    continue;
  }

  // blank
  if (!line.trim()) { i++; continue; }

  // paragraph (join wrapped lines until blank / structural)
  let buf = [line];
  i++;
  while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\s*[-*]\s|\s*\d+\.\s|\s*\|)/.test(lines[i]) && !/^\s*---+\s*$/.test(lines[i])) {
    buf.push(lines[i]); i++;
  }
  const text = buf.join(" ").trim();
  const italicWhole = /^\*[^*].*\*$/.test(text) && !text.includes("**");
  children.push(new Paragraph({
    spacing: { before: 60, after: 140, line: 300 },
    alignment: AlignmentType.JUSTIFIED,
    children: italicWhole
      ? [new TextRun({ text: stripInline(text.slice(1, -1)), font: BODY, size: 20, italics: true, color: "5E7176" })]
      : runs(text),
  }));
}

// ---------- document ----------
const doc = new Document({
  creator: "Ujjwal Thakran",
  title: "MNKAI-III Manuscript",
  description: "Frozen multi-backbone foundation-feature probe with a deployable trust stack for dark-skin dermatology decision support",
  numbering: {
    config: [{
      reference: "num",
      levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.START,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.22) } } },
      }],
    }],
  },
  styles: {
    default: { document: { run: { font: BODY, size: 22 } } },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },                 // US Letter
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          spacing: { after: 120 },
          children: [new TextRun({ text: "MNKAI-III — manuscript draft", font: SANS, size: 16, color: "7A8C90" })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: SANS, size: 16, color: "7A8C90" })],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("WROTE", OUT, "(" + (buf.length / 1024).toFixed(0) + " KB, " + children.length + " blocks)");
});
