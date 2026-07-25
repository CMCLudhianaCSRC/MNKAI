# MOUNT SPEC — 3D-printed lesion spacer (camera standoff)
## North India Dermatology Photo Data Collection Study

Version 1.0 | Companion to `PHOTO_CAPTURE_SOP.md` | Printable source: `lesion_spacer.scad`

---

## 1. What it is, and the one problem it solves

The spacer is a light plastic cone that clips over the phone's **main rear camera** and rests its wide open end on the patient's skin. It does one job:

> **It holds the camera at a fixed, known distance from the skin — every time, for every operator, on every day, on every phone.**

Set by eye, one nurse shoots at 8 cm and another at 25 cm, and the same 6 mm lesion appears at completely different sizes and sharpness. Fixing the distance gives **consistent scale** (a known length on the ruler maps to about the same pixel count on a given phone, so lesion size can be measured afterwards), **consistent focus** (phone main cameras generally cannot focus closer than about **10–15 cm**, and going inside that limit is the commonest cause of soft close-ups), and **consistent framing**.

It is a standardisation aid — not an optical device, and no substitute for the clip-on dermoscope.

---

## 2. Design requirements

| # | Requirement | How the design meets it |
|---|---|---|
| 1 | Fits over the **main** rear camera only | A band slides over the top edge of the phone; a collar and clear hole sit over the main lens. Lens position is a parameter you measure. |
| 2 | Holds the lens at a **set working distance** | `working_distance`, **default 120 mm**, adjustable **100–150 mm**. Below ~100 mm most phones cannot focus. |
| 3 | Ambient light must reach the lesion; the operator must see it | Three large cut-away side windows leaving three ribs. |
| 4 | Colour card and ruler must sit **at the skin plane** | Two flat channels at the skin end let the card and ruler slide in and lie flat on the skin, in the lesion's plane. |
| 5 | Must not press or **blanch** the skin | The contact end is a **wide flat flange**, not an edge, so load is spread. Blanching whitens skin and changes lesion colour. A soft foam or silicone strip is stuck to this flange. |
| 6 | Wipeable between patients | Smooth surfaces, no crevices, no fabric, no sharp edges; two-part push fit so both pieces can be wiped all over. |
| 7 | Printable on a basic printer **without supports** | Two parts, each with a self-supporting orientation (see the `.scad` header). |

---

## 3. Materials

- **PLA** — cheapest and easiest. Caveat: repeated 70% alcohol wiping can craze and whiten it over months.
- **PETG — recommended**, because the protocol is to alcohol-wipe between every patient. Tougher, alcohol-tolerant, needs no enclosure.
- **Avoid ABS** (fumes, warping) and **avoid resin printing** (uncured resin irritates skin).
- Prefer filament sold as **skin-contact/food-contact safe**. A 3D print can never be sterile — layer lines hold organisms. Hence: **intact skin only**, wiped between patients, never on a wound, ulcer or mucosal surface.
- **No sharp edges.** Deburr the flange with fine sandpaper; check by running a finger round it.
- Print with a **brim**; the cone has a large flat base and will lift without one.

---

## 4. Print settings (typical, not exact)

Starting points for a 0.4 mm nozzle. Slicer defaults usually work; treat this as a guide.

| Setting | Cone (standoff) | Clip (phone band) |
|---|---|---|
| Layer height | 0.20–0.28 mm | 0.16–0.20 mm |
| Walls / perimeters | 3 | 3–4 |
| Infill | 10–15 % | 20–25 % |
| Supports | **None** | **None** |
| Brim | Yes | Yes |
| Orientation | Wide (skin) end down | Screen-side face down, collar up |
| Approx. filament | ~40–55 g | ~15–25 g |
| Approx. print time | 4–8 h | 1–2 h |

Prints nearly always fail at the base: clean the bed, use the brim, slow the first layer.

---

## 5. Cost

| Item | Typical cost | Notes |
|---|---|---|
| Filament, **self-printed**, per unit | **Rs 30–100** | Cone plus clip sit near the upper end. |
| **Print service**, per part | **Rs 150–400** | Two parts per unit. |
| **Entry-level 3D printer**, one-time | **Rs 12–18 k** | |

**Recommendation:** the study needs roughly **8–10 working units** across two sites, plus iterations while the fit is dialled in per phone model. At print-service rates that already exceeds the cost of an entry-level printer, after which each reprint costs only filament — so **buying the printer is the better value here**. Decide before ordering, not after.

For scale: the **clip-on dermoscope costs Rs 5–15 k** and is the item the cancer arm actually depends on. The spacer must never be funded at its expense.

---

## 6. Fitting it to a specific phone

Once per phone model, **case off**, with a vernier caliper.

1. **Find the main lens.** Open the camera at `1x` and cover each lens in turn — the one that blacks out the preview is the main lens. Never set up on the ultrawide or the small 2 MP macro lens.
2. **Measure and write down:** `phone_width`, `phone_thickness`, `camera_offset_x` and `camera_offset_y` (from the left and top edges of the back to the main lens centre), `lens_bump` (how far the lens stands above the flat back), the camera-island size and position, and `aperture_diameter` (lens barrel outer diameter **plus about 6 mm** — too small and a dark ring appears at the edge of every photo).
3. **Edit the parameter block at the top of `lesion_spacer.scad`.** Every parameter carries a comment saying what to measure.
4. **Print the 10 mm test ring first** (`part = "testring"`; about 10 minutes, a few rupees). It should slide on with light finger pressure and stay put. Too tight: raise `fit` to 0.5–0.6. Too loose: drop it to 0.3. **Do not print the full parts until the test ring fits.**
5. Print `part = "clip"`, then `part = "cone"`. Push the collar into the socket — firm, no rocking. Adjust `fit_joint` and reprint the clip if needed.
6. **Label the unit** with phone model and working distance (e.g. `120 mm — Redmi Note 13`). Units are not interchangeable between phone models.

---

## 7. Calibration and validation — before the unit enters clinic

**Repeat for every phone in the study.**

1. Fit the spacer and photograph a **ruler lying flat**, exactly as the SOP describes (main lens, `1x`, no zoom, default photo mode, focus and exposure locked, flash off).
2. On a computer, count how many pixels a known length (say 10 mm) spans. Record **millimetres per pixel** for that phone.
3. **Repeat three times**, refitting the spacer each time. The readings should agree closely; if they do not, the clip is loose or the flange is not sitting flat — fix that first.
4. Record the value, phone model, working distance and date in the equipment log. **Re-check every 3 months** and after any repair or reprint.
5. **Sanity-check the framing:** confirm a typical lesion still gives at least **1500 pixels across the lesion region** and fills at least half the frame. If not, shorten the working distance towards 100 mm and re-calibrate, or accept a wider close-up and rely on the dermoscopic shot for surface detail.

For context, **HAM10000** — the gold-standard dermoscopy benchmark — is distributed at just **600 × 450 pixels**. Resolution is not the bottleneck; focus, framing and lighting are.

---

## 8. Cleaning and disinfection

- **Between every patient:** wipe the whole spacer, especially the skin-contact flange, with a 70% alcohol wipe and let it **air-dry fully** before reuse.
- **Between sessions:** separate the two parts and wipe both, including inside the cone.
- **Weekly:** mild soap and lukewarm water, rinse, air-dry. Never hot water, a dishwasher, or a car dashboard — the plastic deforms and the working distance is no longer what the label says.
- **Replace** if cracked, crazed, stained, warped, sharp-edged, or if the foam strip will not stay stuck. A reprint costs a few tens of rupees; a damaged unit costs data.
- **Never** use it on broken skin, an ulcer, a weeping lesion, a mucosal surface, or a patient in isolation precautions. Photograph hand-held instead and note it.

---

## 9. Limitations and notes

**Read this before presenting the device to the Ethics Committee.**

1. **This is a research aid, not a medical device.** It is a **non-sterile accessory used on intact skin only**. It makes no diagnosis, produces no reading, and has no effect on treatment. It is not registered or regulated as a medical device and must not be described as one in any document, consent form or presentation.
2. **A 3D print cannot be sterilised.** Alcohol wiping between patients is decontamination appropriate to intact-skin contact — nothing more.
3. **The cancer arm depends on the dermoscope, not the spacer.** The biopsy-confirmed cancer images are what make this dataset distinct, and their detail comes from the **clip-on dermoscope**. If a budget choice must be made, the dermoscope wins.
4. **The wide-field trade-off is real.** At 120 mm a phone main camera sees a fairly wide area of skin, and the cone's 90 mm opening crops that to a fixed circle — good for standardisation, but a small lesion will still not fill half the frame. *Recommendation, not established fact:* test 100, 120 and 150 mm on each phone, choose the shortest distance at which it focuses reliably, and fix it for the study. For small lesions the dermoscopic image carries the surface detail.
5. **Units are phone-specific.** The parameters, and the millimetres-per-pixel figure, are valid only for the phone they were measured on.
6. **It fixes distance and tilt, nothing else.** All of `PHOTO_CAPTURE_SOP.md` still applies — especially the dark-skin exposure check.
7. **Patient comfort overrides the protocol.** If the flange is uncomfortable or the patient objects, photograph hand-held and record that.
8. **Record what was used** in every log entry: spacer yes/no, which unit, what working distance.

---

*Open `lesion_spacer.scad` in OpenSCAD 2019.05 or newer, edit the parameter block, press F6, then File → Export → STL. Questions or a failed fit: contact the study coordinator before printing in quantity.*
