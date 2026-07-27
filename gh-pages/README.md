# MNKAI-III — Data Collection App (demonstration)

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
