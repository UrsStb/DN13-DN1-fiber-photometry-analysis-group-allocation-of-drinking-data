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

The treatment-group allocation script requires R and the following
packages:

readxl
writexl


##Input data

The analysis pipeline expects:

Doric fiber-photometry exports in CSV format.
Behavioural scoring files in XLSX format containing manually scored
HTR frame indices and optional scorer annotations.

File paths and experimental conditions are specified in the configuration
section of the group-analysis script [or: in the accompanying manifest,
if you moved them to a separate manifest file].

Raw experimental data are not included in this repository.

Fiber-photometry analysis

Recordings contain a 405-nm reference channel and a 465/470-nm
glutamate-sensitive iGluSnFR channel.

The main analysis includes:

interpolation of the 405-nm reference onto signal timestamps
3-Hz second-order Butterworth low-pass filtering
robust reference fitting using Tukey biweight regression
reference-based ΔF/F calculation
HTR alignment using recorded synchronization TTL pulses
extraction of event epochs from -8 to +4 s relative to HTR onset
event-specific baseline normalization using -5 to -2 s
primary pre-HTR analysis from -2.0 to -0.5 s
calculation of peak amplitude, peak latency, signed AUC, and positive AUC
averaging of events within sessions and animals before group-level analysis

The animal constitutes the biological replicate for group-level
comparisons.

Quality control

Individual HTR events are excluded when:

another HTR occurs within the predefined analysis span,
the complete analysis interval lies outside the recording, or
the event was flagged during behavioural scoring as questionable.

The same retained event set is used across outcome measures.

fp_session_qc.py additionally provides recording-level diagnostics
before event-related analysis.

##Usage

Run session quality control first:

python fp_session_qc.py

Raw fluorescence traces can be inspected using:

python inspect_raw_traces.py

The main HTR-aligned group analysis is run using:

python group_peri_event_analysis.py

htr_sync.py must be located in the same Python module directory as
group_peri_event_analysis.py.

Input paths and experimental settings should be adjusted in the
configuration section before running the scripts.

##Outputs

Depending on the selected configuration, the analysis produces:

animal-level and condition-level response summaries
HTR counts and exclusion information
Prism-ready Excel output
event-related fluorescence figures
reference-channel control plots
Data availability

Raw photometry recordings, behavioural videos, scoring files, and
animal-level source data are not publicly included in this repository.

Access to underlying experimental data is subject to the policies and
permissions of the originating research institution.

## Code availability

The code in this repository corresponds to the analysis pipeline used
for the associated Master's thesis.

The version used for the final thesis analyses is archived as release
v1.0.

The analysis code is additionally available from the author upon
reasonable request.

## Citation

If you use or adapt this code, please cite the associated Master's thesis:

[Steinberger, U]. [2026]. The mGluR2 positive allosteric modulator DN13-DN1
attenuates early relapse-like ethanol drinking in the alcohol deprivation
effect model. Master's thesis, Heidelberg University.

License

No license has currently been assigned. Please contact the author before
reuse or redistribution.
