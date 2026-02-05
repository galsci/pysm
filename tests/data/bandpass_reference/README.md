# Reference Bandpass Files from Simons Observatory

These files are reference outputs from the original bandpass sampling implementation used in MBS 16, sourced from: https://github.com/simonsobs/bandpass_sampler

## Files

### LAT (Large Aperture Telescope)
- `LAT_LF1_w0_reference.tbl` - Low Frequency band 1, Wafer 0 (~27 GHz)
- `LAT_LF1_w1_reference.tbl` - Low Frequency band 1, Wafer 1 (~27 GHz)
- `LAT_HF2_w0_reference.tbl` - High Frequency band 2, Wafer 0 (~280 GHz)
- `LAT_HF2_w1_reference.tbl` - High Frequency band 2, Wafer 1 (~280 GHz)

### SAT (Small Aperture Telescope)
- `SAT_LF1_w0_reference.tbl` - Low Frequency band 1, Wafer 0 (~27 GHz)
- `SAT_LF1_w1_reference.tbl` - Low Frequency band 1, Wafer 1 (~27 GHz)
- `SAT_HF2_w0_reference.tbl` - High Frequency band 2, Wafer 0 (~280 GHz)
- `SAT_HF2_w1_reference.tbl` - High Frequency band 2, Wafer 1 (~280 GHz)

## File Format

IPAC table format with two columns:
- `bandpass_frequency` (GHz) - Frequency values
- `bandpass_weight` - Normalized bandpass transmission weights

## Purpose

These files serve as validation references to ensure the PySM bandpass sampling implementation produces results consistent with the original MBS 16 approach.

## License

These files are from the simonsobs/bandpass_sampler repository and are subject to its license terms (BSD-3-Clause).
