"""Generate the flow diagrams for the research-project submission.

Black-and-white, Times New Roman, sized for A4 with 2 cm margins. Written with PIL so
the figures regenerate with the document and need no external drawing tool.

    python scripts/_make_protocol_figures.py
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "reports" / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TIMES = "C:/Windows/Fonts/times.ttf"
TIMESB = "C:/Windows/Fonts/timesbd.ttf"
S = 2                       # supersampling factor for crisp text
BLACK, GREY, LGREY, WHITE = (0, 0, 0), (90, 90, 90), (215, 215, 215), (255, 255, 255)


def F(size, bold=False):
    return ImageFont.truetype(TIMESB if bold else TIMES, size * S)


def wrap(draw, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= maxw:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def box(d, x, y, w, h, text, *, bold=False, fill=WHITE, size=11, radius=6, dashed=False):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], radius=radius * S,
                        fill=fill, outline=BLACK, width=2)
    f = F(size, bold)
    lines = wrap(d, text, f, (w - 12) * S)
    lh = (size + 3) * S
    ty = (y + h / 2) * S - len(lines) * lh / 2
    for ln in lines:
        tw = d.textlength(ln, font=f)
        d.text(((x + w / 2) * S - tw / 2, ty), ln, font=f, fill=BLACK)
        ty += lh


def arrow(d, x1, y1, x2, y2, head=7):
    d.line([x1 * S, y1 * S, x2 * S, y2 * S], fill=BLACK, width=2)
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    for s in (+1, -1):
        d.line([x2 * S, y2 * S,
                (x2 - head * math.cos(a - s * 0.5)) * S,
                (y2 - head * math.sin(a - s * 0.5)) * S], fill=BLACK, width=2)


def label(d, x, y, text, size=9, bold=False, anchor="lt", italic=False):
    f = F(size, bold)
    tw = d.textlength(text, font=f)
    px = x * S - (tw / 2 if anchor == "mt" else 0)
    d.text((px, y * S), text, font=f, fill=BLACK if bold else GREY)


def canvas(w, h):
    im = Image.new("RGB", (w * S, h * S), WHITE)
    return im, ImageDraw.Draw(im)


def save(im, name, w):
    im = im.resize((w * S // 2 * 2 // S * 2, im.height // S * 2), Image.LANCZOS)
    im.save(OUT / name, dpi=(300, 300))
    print("  wrote", name, im.size)


# ============================ FIGURE 1 — master study flow ============================
def fig1():
    W, H = 620, 850          # content runs to ~825; keep headroom for the final box
    im, d = canvas(W, H)
    f = F(12, True)
    d.text((10 * S, 6 * S), "Figure 1. Overall flow of the study", font=f, fill=BLACK)

    cx = 310
    y = 40
    box(d, cx - 140, y, 280, 34, "Institutional Ethics Committee approval", bold=True)
    arrow(d, cx, y + 34, cx, y + 56); y += 56

    box(d, cx - 140, y, 280, 34, "Month 0: set-up, application build, staff briefing")
    arrow(d, cx, y + 34, cx, y + 56); y += 56

    # split into two arms
    box(d, cx - 140, y, 280, 30, "Patient attends Dermatology OPD", bold=True)
    arrow(d, cx, y + 30, cx, y + 52); y += 52

    box(d, cx - 140, y, 280, 30, "Eligibility screened against criteria")
    arrow(d, cx, y + 30, cx, y + 52); y += 52

    box(d, cx - 140, y, 280, 30, "Written informed consent taken in the application")
    arrow(d, cx, y + 30, cx, y + 52); y += 52

    box(d, cx - 140, y, 280, 30, "Clinical details and photographs recorded")
    arrow(d, cx, y + 30, cx, y + 56); y += 56

    # the blinding step - most important box
    box(d, cx - 175, y, 350, 40,
        "Dermatologist records diagnosis FIRST, blinded to the system output",
        bold=True, fill=LGREY)
    arrow(d, cx, y + 40, cx, y + 62); y += 62

    box(d, cx - 140, y, 280, 30, "System output recorded separately")
    # branch
    ybr = y + 30
    arrow(d, cx, ybr, cx, ybr + 24)
    y = ybr + 24

    d.line([(cx - 150) * S, y * S, (cx + 150) * S, y * S], fill=BLACK, width=2)
    arrow(d, cx - 150, y, cx - 150, y + 22)
    arrow(d, cx + 150, y, cx + 150, y + 22)
    y += 22

    box(d, cx - 260, y, 220, 44, "Objective 1: first 100 patients — validation against dermatologist")
    box(d, cx + 40, y, 220, 44, "Objective 2: all consenting patients over 24 months — image repository")
    y += 44

    arrow(d, cx - 150, y, cx - 150, y + 26)
    arrow(d, cx + 150, y, cx + 150, y + 26)
    y += 26

    box(d, cx - 260, y, 220, 34, "Biopsy if clinically indicated (routine care)")
    box(d, cx + 40, y, 220, 34, "Two dermatologists label; third adjudicates")
    y += 34

    arrow(d, cx - 150, y, cx - 150, y + 26)
    arrow(d, cx + 150, y, cx + 150, y + 26)
    y += 26

    box(d, cx - 260, y, 220, 34, "Histopathology linked to study number")
    box(d, cx + 40, y, 220, 34, "De-identified images stored on institutional server")
    y += 34

    # merge
    arrow(d, cx - 150, y, cx - 150, y + 20)
    arrow(d, cx + 150, y, cx + 150, y + 20)
    y += 20
    d.line([(cx - 150) * S, y * S, (cx + 150) * S, y * S], fill=BLACK, width=2)
    arrow(d, cx, y, cx, y + 24); y += 24

    box(d, cx - 140, y, 280, 30, "Analysis as specified in the protocol")
    arrow(d, cx, y + 30, cx, y + 52); y += 52
    box(d, cx - 140, y, 280, 32, "Publication of both objectives", bold=True)

    label(d, 14, 300, "Months 1–24", size=10, bold=True)
    save(im, "fig1_study_flow.png", W)


# ==================== FIGURE 2 — what happens at one OPD visit ====================
def fig2():
    W, H = 620, 150
    im, d = canvas(W, H)
    d.text((10 * S, 6 * S), "Figure 2. Sequence at a single outpatient visit (5–10 minutes)",
           font=F(12, True), fill=BLACK)
    steps = ["Eligibility checked", "Information sheet\nand consent",
             "Clinical details\nentered", "Photographs\ntaken",
             "Dermatologist's\ndiagnosis recorded", "System run,\noutput stored"]
    x, y, w, h, gap = 12, 46, 92, 60, 8
    for i, s in enumerate(steps):
        fill = LGREY if i == 4 else WHITE
        box(d, x, y, w, h, s.replace("\n", " "), size=10, fill=fill)
        if i < len(steps) - 1:
            arrow(d, x + w, y + h / 2, x + w + gap, y + h / 2)
        x += w + gap
    label(d, 12, 116, "The shaded step is recorded before the system is run and while blinded to it.",
          size=9)
    save(im, "fig2_visit_sequence.png", W)


# ==================== FIGURE 3 — how the system produces an output ====================
def fig3():
    W, H = 620, 250
    im, d = canvas(W, H)
    d.text((10 * S, 6 * S), "Figure 3. How the system produces its output", font=F(12, True), fill=BLACK)

    box(d, 14, 50, 88, 46, "Photograph", bold=True)
    arrow(d, 102, 73, 124, 73)

    # six frozen models
    d.rounded_rectangle([124 * S, 36 * S, 250 * S, 128 * S], radius=6 * S, outline=BLACK, width=2)
    label(d, 187, 40, "Six frozen foundation models", size=9, bold=True, anchor="mt")
    names = ["DINOv2-B", "DINOv2-reg", "DINOv2-L", "EVA-02", "ConvNeXt-V2", "SigLIP"]
    for i, n in enumerate(names):
        bx, by = 130 + (i % 2) * 60, 54 + (i // 2) * 24
        box(d, bx, by, 56, 18, n, size=8)
    arrow(d, 250, 82, 272, 82)

    box(d, 272, 62, 78, 40, "Ensemble: scores averaged", size=10)
    arrow(d, 350, 82, 372, 82)
    box(d, 372, 62, 78, 40, "Calibration of probability", size=10)
    arrow(d, 450, 82, 472, 82)

    box(d, 472, 40, 134, 34, "Confident: ranked list of diagnoses + probability of malignancy", size=9)
    box(d, 472, 84, 134, 34, "Not confident: declines and advises referral", size=9, fill=LGREY)
    d.line([466 * S, 57 * S, 466 * S, 101 * S], fill=BLACK, width=2)
    arrow(d, 466, 57, 472, 57)
    arrow(d, 466, 101, 472, 101)

    label(d, 14, 150, "The six models are used unchanged (frozen); only a small classifier is trained on the", size=9)
    label(d, 14, 165, "feature vectors they produce. Averaging independent models is termed ensembling.", size=9)
    label(d, 14, 185, "Calibration makes the stated probability match observed accuracy; the abstention rule", size=9)
    label(d, 14, 200, "(shaded) is what allows the system to decline rather than guess.", size=9)
    save(im, "fig3_system.png", W)


# ==================== FIGURE 4 — the data collection application ====================
def fig4():
    W, H = 620, 290
    im, d = canvas(W, H)
    d.text((10 * S, 6 * S), "Figure 4. Screens of the data collection application",
           font=F(12, True), fill=BLACK)

    screens = [
        ("1. Consent", ["Patient information", "sheet displayed in", "English / Hindi /", "Punjabi", "", "Signature captured", "on screen"]),
        ("2. Clinical details", ["Age, sex, Fitzpatrick", "type", "Site, duration,", "symptoms, evolution", "", "Controlled lists,", "no free text"]),
        ("3. Photograph", ["Framing guide shown", "", "Blur and exposure", "checked before save", "", "Retake prompted if", "inadequate"]),
        ("4. Storage", ["Study number only", "", "No name, no face", "", "Encrypted transfer to", "institutional server", "on the same day"]),
    ]
    x, y, pw, ph = 22, 44, 130, 200
    for title, lines in screens:
        # phone outline
        d.rounded_rectangle([x * S, y * S, (x + pw) * S, (y + ph) * S], radius=12 * S,
                            outline=BLACK, width=2, fill=WHITE)
        d.rounded_rectangle([(x + 8) * S, (y + 22) * S, (x + pw - 8) * S, (y + ph - 14) * S],
                            radius=4 * S, outline=GREY, width=1)
        d.line([(x + pw / 2 - 12) * S, (y + 12) * S, (x + pw / 2 + 12) * S, (y + 12) * S],
               fill=GREY, width=3)
        f = F(10, True)
        tw = d.textlength(title, font=f)
        d.text(((x + pw / 2) * S - tw / 2, (y + 30) * S), title, font=f, fill=BLACK)
        fy = y + 50
        for ln in lines:
            if ln:
                fs = F(9)
                lw = d.textlength(ln, font=fs)
                d.text(((x + pw / 2) * S - lw / 2, fy * S), ln, font=fs, fill=BLACK)
            fy += 13
        if x + pw + 12 < W - 40:
            arrow(d, x + pw + 2, y + ph / 2, x + pw + 14, y + ph / 2)
        x += pw + 18

    label(d, 22, 256, "Consent is taken before the camera opens. The application will not proceed to the", size=9)
    label(d, 22, 270, "photograph step until consent has been recorded.", size=9)
    save(im, "fig4_app.png", W)


# ==================== FIGURE 5 — histopathology linkage ====================
def fig5():
    W, H = 620, 170
    im, d = canvas(W, H)
    d.text((10 * S, 6 * S), "Figure 5. Linking a biopsy result to the study record",
           font=F(12, True), fill=BLACK)
    box(d, 14, 48, 112, 44, "Biopsy taken on clinical indication", size=10)
    arrow(d, 126, 70, 146, 70)
    box(d, 146, 48, 112, 44, "Histopathology number written on proforma", size=10)
    arrow(d, 258, 70, 278, 70)
    box(d, 278, 48, 112, 44, "Case entered in pending register", size=10, fill=LGREY)
    arrow(d, 390, 70, 410, 70)
    box(d, 410, 48, 96, 44, "Report returns (weeks later)", size=10)
    arrow(d, 506, 70, 524, 70)
    box(d, 524, 48, 82, 44, "Diagnosis entered; grade updated", size=9)
    label(d, 14, 108, "Only after this step does a case count as histopathologically confirmed. Cases whose", size=9)
    label(d, 14, 122, "report cannot be traced keep their clinical diagnosis and are reported separately.", size=9)
    save(im, "fig5_histopath.png", W)


if __name__ == "__main__":
    print("generating figures ->", OUT)
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("done")
