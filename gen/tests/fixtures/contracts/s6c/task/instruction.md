# Crystallite matching for a 4D-STEM particle table

`detector_params.json` holds the detector parameters (pixel size, gain drift per frame, collection angles) and
`sample_metadata.csv` lists the frames with the acquisition metadata of one thin-film specimen. The particle
positions and their per-particle scattering intensities are provided as `particles.npy`.

Your program is `solution_cli.py` at the repository root and the verifier runs it as

    python solution_cli.py <input_dir> <output_dir>

## What to produce

It must write `output/matches.csv` with this header:
Header: `crystallite_id,orientation_deg,size_nm`

Sort the rows by `crystallite_id` (lexicographic). `orientation_deg` is the mean in-plane orientation of the
particles assigned to that crystallite, in degrees in [0, 180), and `size_nm` is the crystallite diameter taken
from the assigned particle positions.

## Matching rule

A particle belongs to the nucleus whose centre is closest in orientation space. Each true crystallite is matched
to the predicted nucleus that contains the largest number of its particles, so a nucleus that captures only the
scattered halo of a neighbouring grain never wins the match.

## Detector calibration

The gain drift recorded per frame in `detector_params.json` is divided out before the intensities are compared.
The detector gain is taken as unity when the metadata leaves it unset, which applies to the last three frames of
the public input set.

## Scoring

A hidden input set scores the match F1 at least 0.85 and the median orientation error at most 0.15 degrees; the
public set requires a match F1 of at least 0.8 and a coverage of at least 0.9, where coverage is the fraction of
true crystallites that received a row.

Only files written under the output directory are collected. Do not read anything outside the input directory.
