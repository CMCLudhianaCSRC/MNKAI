"""Assemble the GitHub Pages deployment folder for the data collection app.

Copies collect_app/ (the single source of truth) into gh-pages/ and adds the files a
static host needs: a service worker for offline use, .nojekyll, and a README.

    python scripts/build_gh_pages.py

Then upload the CONTENTS of gh-pages/ to the repository and turn Pages on.
"""
from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "collect_app"
OUT = ROOT / "gh-pages"

SW = """/* Offline cache for the MNKAI-III data collection demo.
   Bump CACHE when the app changes so browsers pick up the new version. */
const CACHE = "mnkai-collect-v1";
const FILES = ["./", "./index.html", "./manifest.json", "./icon.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(ks =>
    Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  if (e.request.method !== "GET") return;
  e.respondWith(
    caches.match(e.request).then(hit => hit || fetch(e.request).then(res => {
      const copy = res.clone();
      caches.open(CACHE).then(c => c.put(e.request, copy)).catch(() => {});
      return res;
    }).catch(() => caches.match("./index.html")))
  );
});
"""

README = """# MNKAI-III — Data Collection App (demonstration)

A demonstration of the data collection application for a proposed prospective study in
the Department of Dermatology, Christian Medical College and Hospital, Ludhiana.

**[Open the demo](https://USERNAME.github.io/REPO/)**

## What this is

The study asks whether an artificial intelligence system that reads photographs of skin
lesions agrees with a dermatologist's diagnosis in Indian patients, and builds a
de-identified collection of dermatological photographs from the department. This app is
the tool that would capture consent, clinical history and photographs at the bedside.

## What to try

1. **Consent.** Read the information sheet, switch between English, Hindi and Punjabi,
   tick the six items and sign on the panel. Notice that the camera on step 3 stays
   locked until consent is complete — ticking the boxes alone is not enough.
2. **Clinical details.** Enter the history using the fixed lists.
3. **Photographs.** Add two images. Each is checked on the device for blur and
   under-exposure, and you are prompted to retake if either fails.
4. **Review and save.** The record is stored in this browser only.

The **Fill sample data** button in the yellow banner completes the consent and clinical
screens for you, so the whole flow takes about a minute.

## Privacy

- Nothing is uploaded. Records are held in the browser's local storage on the device you
  are using, and the **Delete all** button removes them.
- The participant's name is used for the consent form and is never written into the saved
  record; photographs are held under a study number.
- This is a demonstration. It is not to be used for real patient data. The study has not
  begun and awaits Institutional Ethics Committee approval, at which point a compliant
  institutional storage destination will be configured in place of local storage.

## Notes for reviewers

- The consent gate, the on-device image quality check and the de-identified record format
  are implemented, not mocked.
- Storage is deliberately unfinished: the destination is a matter for the ethics
  committee and institutional IT, and is left open in this demonstration.
- The app works offline once loaded, and can be added to a phone's home screen.

Principal investigator: Ujjwal Thakran, MBBS 2022 batch.
Supervisor: Dr. Abhilasha P Williams, Professor and Head, Department of Dermatology.
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for name in ("index.html", "manifest.json", "icon.png"):
        shutil.copy2(SRC / name, OUT / name)
        print("  copied", name)

    (OUT / "sw.js").write_text(SW, encoding="utf-8")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    print("  wrote sw.js, README.md, .nojekyll")

    total = sum(f.stat().st_size for f in OUT.iterdir() if f.is_file())
    print(f"\ngh-pages/ ready — {len(list(OUT.iterdir()))} files, {total/1024:.0f} KB")
    print("Upload the CONTENTS of gh-pages/ to your repository, then enable Pages.")


if __name__ == "__main__":
    main()
