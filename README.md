# DN13-DN1-fiber-photometry-analysis-group-allocation-of-drinking-data
Analysis code associated with my Master's Thesis investigating the brain-penetrant mGluR2 positive allosteric modulator DN13-DN1.

##Repository contents
### Fiber photometry
- `group_peri_event_analysis.py`
  Main event-related fiber-photometry analysis.
- `htr_sync.py`
  Synchronization of manually scored HTR video frames with photometry timestamps.
- `fp_session_qc.py`
  Quality-control checks for individual Doric fiber-photometry recordings.
- `inspect_raw_traces.py`
  Visualization and inspection of unprocessed 405-nm and 465/470-nm fluorescence traces.

### Group allocation
- `group_allocation.R`
  Covariate-balancing procedure used for treatment-group allocation.

## Analysis environment
Python 3.14.7 was used for the final fiber-photometry analyses.

## Data availability
Raw photometry recordings, behavioural videos, and animal-level source data
are not included in this repository.

## Code availability
The code provided here corresponds to the analyses performed for the
associated Master's thesis.

