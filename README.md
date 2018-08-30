# annotation_compute
A music feature Dataset generated by librosa and madmom package in Python for MOST Grant Proposal- Personalized Music Health Care Assisted by Deep Machine Learning

## Overview of the Proposal dataset

See https://github.com/DennyHsieh/music-health-dataset for a complete reference. (In Chinese)


## Description
- Files naming
  - Files in “annotation_file_librosa” folder are the features computing by librosa package in Python.
  - Files in “annotation_file_madmom” folder are the features computing by madmom package in Python.
  - File names are named by “song_id-feature.npy”
- Feature description


| **Name**                 | **Description**                                              |
| ------------------------ | ------------------------------------------------------------ |
| ***song_id***            | song ID for each music in MHCDL Dataset                      |
| ***kkbox_id***           | KKBOX ID for each music in KKBOX                             |
| ***librosa_beat_times*** | start time of each beat (in BPM) computing by librosa        |
| ***librosa_chromagram*** | chroma features for each segment computing by librosa        |
| ***librosa_duration***   | duration of the track (in seconds) computing by librosa      |
| ***librosa_mfcc***       | MFCC features computing by librosa                           |
| ***librosa_onset***      | beginning of a musical note (in second) computing by librosa |
| ***librosa_sr***         | sampling rate (in Hz) computing by librosa                   |
| ***librosa_tempo***      | tempo (in BPM) computing by librosa                          |
| ***madmom_onset***       | beginning of a musical note (in second) computing by madmom  |



## Reference
1. https://github.com/CPJKU/madmom
2. https://github.com/librosa/librosa

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.591533.svg)](https://doi.org/10.5281/zenodo.591533)


## Acknowledgements

This research is partially supported by the Program of Ministry of Science and Technology, Taiwan, R.O.C. under Grant no. MOST 106-3114-E-007-013.
本研究感謝科技部「數位經濟前瞻技術研發與應用專案計畫」(MOST 106-3114-E-007-013)

