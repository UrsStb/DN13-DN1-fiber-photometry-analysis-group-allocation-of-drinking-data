# DN13-DN1 thesis analysis

Analysis code associated with a Master's thesis investigating the
brain-penetrant mGluR2 positive allosteric modulator DN13-DN1 in a
rodent model of relapse-like alcohol drinking.

The repository contains the code used for treatment-group allocation,
fiber-photometry quality control, synchronization of manually scored
head-twitch responses (HTRs) with photometry recordings, and
event-related fiber-photometry analysis.

## Repository contents

### Fiber photometry

- `group_peri_event_analysis.py`  
  Main group-level event-related analysis of iGluSnFR fiber-photometry
  recordings. HTR-aligned signals are processed and aggregated from
  individual events to sessions and subsequently to animal × condition
  values for group-level analysis.

- `htr_sync.py`  
  Maps manually scored video-frame indices to photometry timestamps
  using the recorded synchronization TTL train.

- `fp_session_qc.py`  
  Quality-control utility for individual Doric fiber-photometry
  recordings, including checks of timebase integrity, channel quality,
  synchronization signals, and optional video integrity.

- `inspect_raw_traces.py`  
  Displays unprocessed 405-nm reference and 465/470-nm
  glutamate-sensitive fluorescence traces prior to preprocessing.

### Treatment-group allocation

- `group_allocation.R`  
  Covariate-balancing procedure used for treatment-group allocation
  based on baseline ethanol intake and body weight.

## Software environment

The final fiber-photometry analyses were performed using Python 3.14.7.

Required Python packages are listed in `requirements.txt`.

Install them using:

```bash
pip install -r requirements.txt
```

The treatment-group allocation script requires R and the following
packages:

- `readxl`
- `writexl`


## Input data

The analysis pipeline expects:

1. Doric fiber-photometry exports in CSV format.
2. Behavioural scoring files in XLSX format containing manually scored HTR frame indices and optional scorer annotations.

File paths and experimental conditions are specified in the configuration
section of the group-analysis script.

Raw experimental data are not included in this repository.

