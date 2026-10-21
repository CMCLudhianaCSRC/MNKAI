# MNKAI-III · web demonstration

Two demonstrations for a proposed prospective study in the Department of Dermatology, Christian Medical
College and Hospital, Ludhiana. The study asks whether an artificial intelligence system that reads photographs
of skin lesions agrees with a dermatologist's diagnosis in Indian patients, and builds a de-identified collection
of dermatological images from the department.

## 1. The AI system (web demo) — `index.html`

The real application's screens, with five sample photographs. The answers shown were produced by MNKAI-III v1.0
on the study laptop and recorded; this page replays them exactly as the laptop shows them, including the
analysis steps, heat maps, the possible-diagnoses list, the disease group and the two-level cancer check.

- No model runs in the browser and nothing is uploaded. Uploading your own photo is only available on the live
  version (the study laptop); here you select one of the samples, and its patient context is filled in for you.
- The five examples are clear cases the system gets right, chosen to show what the screens look like:
  psoriasis and crusted scabies on dark skin, tinea on a smartphone photo from a clinic in Bangladesh, and two
  cancers that raise the red flag, an acral melanoma on the heel and a basal cell carcinoma. They are not a
  measure of accuracy: the Evidence section gives that, and on clinic photographs from South Asia the system is
  right far less often. All five come from test sets the system was never trained on, and the two cancers were
  not used to train its cancer check either.
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

Sample photographs: [psoriasis](http://atlasdermatologico.com.br/img?imageId=6300), [scabies](http://atlasdermatologico.com.br/img?imageId=4909), [melanoma (acral)](http://atlasdermatologico.com.br/img?imageId=4275), [bcc](http://atlasdermatologico.com.br/img?imageId=645) from Atlas Dermatológico (atlasdermatologico.com.br), via the Fitzpatrick17k
dataset, licensed [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/); tinea corporis from
[SkinDisNet](https://data.mendeley.com/datasets/yj3md44hxg) (Sultana et al., Data in Brief 2025), licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The processed images and heat maps shown for them
are derived from these photographs and shared under the same licences.

The system was trained on public dermatology image collections. Research and education only; not a diagnostic
device.

Principal investigator: Ujjwal Thakran, MBBS 2022 batch.
Supervisor: Dr Abhilasha P Williams, Professor and Head, Department of Dermatology.

Built by `scripts/record_pages_demo.py` and `scripts/build_pages_site.py` in the MNKAI-III repository.
