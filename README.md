# MNKAI-III · web demonstration

Two demonstrations for a proposed prospective study in the Department of Dermatology, Christian Medical
College and Hospital, Ludhiana. The study asks whether an artificial intelligence system that reads photographs
of skin lesions agrees with a dermatologist's diagnosis in Indian patients, and builds a de-identified collection
of dermatological images from the department.

## 1. The AI system (web demo) — `index.html`

The real application's screens, with three sample photographs. The answers shown were produced by MNKAI-III v1.0
on the study laptop and recorded; this page replays them exactly as the laptop shows them, including the
analysis steps, heat maps, the possible-diagnoses list, the disease group and the two-level cancer check.

- No model runs in the browser and nothing is uploaded. Your own photographs cannot be analysed in this
  version; the system needs the study laptop.
- The three examples were chosen to show the range of behaviour, not the best results: a confident correct
  answer (lichen planus), a correct answer the system was not sure of (vitiligo), and a wrong answer on which
  it abstained but still named the right disease group (scabies).
- The optional Gemini follow-up questions are switched off here; on the laptop they send the photograph to
  Google, so they are never used for study patients.

## 2. The data collection app — `collect/`

The phone app used at the bedside, fully working: information sheet in English, Hindi and Punjabi, the consent
checklist with the separate choice on future use, clinical details from fixed lists, guided photographs with an
on-device quality check, and review and save.

- Nothing is uploaded. Records are held in the browser on the device you are using, and **Delete all** removes
  them. Browser storage is small (about 5 MB), so only a few records with photographs fit; **Export all** first.
  Study numbers keep counting after **Delete all**, so a number is never reused on the same device. The **Fill sample data** button completes the consent and clinical screens for a quick try.
- This is a demonstration. Do not enter real patient data. The study awaits Institutional Ethics Committee
  approval.

## Credits and licence

Sample photographs ([vitiligo](http://atlasdermatologico.com.br/img?imageId=7903), [lichen planus](http://atlasdermatologico.com.br/img?imageId=3826), [scabies](http://atlasdermatologico.com.br/img?imageId=6613)) from Atlas Dermatológico (atlasdermatologico.com.br), via the Fitzpatrick17k
dataset, licensed [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). The processed images and
heat maps shown for them are derived from these photographs and shared under the same licence. All three are
from the outside test set and were never used to train the system.

The system was trained on public dermatology image collections; the licences of some of them, including images
from dermnet.com, are being clarified with their owners, as stated in the study protocol. For that reason this
public demo shows only the outside-test results, which contain no such images. Research and education only; not
a diagnostic device.

Principal investigator: Ujjwal Thakran, MBBS 2022 batch.
Supervisor: Dr Abhilasha P Williams, Professor and Head, Department of Dermatology.

Built by `scripts/record_pages_demo.py` and `scripts/build_pages_site.py` in the MNKAI-III repository.
