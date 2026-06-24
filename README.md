# LDPC Codes Implementation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Course](https://img.shields.io/badge/Course-Information%20Theory-green)
![Topic](https://img.shields.io/badge/Topic-LDPC%20Codes-purple)
![Status](https://img.shields.io/badge/Scope-Coursework%20Simulation-lightgrey)

## Project Overview

This repository implements and evaluates a large finite-length Low-Density Parity-Check (LDPC) code for an Information Theory project.

The main workflow targets a length-200 design request and resolves it to the valid Gallager-regular block length:

```text
(N, K) = (204, 104)
M = 100 parity checks
R = K / N ~= 0.5098
```

The project studies this `(204,104)` LDPC code over BSC, AWGN, and BEC channels using Monte Carlo simulation. It reports BER, FER, convergence rate, average decoder iterations, and signed coding gain. The results are coursework-level experimental results, not capacity-achieving claims.

## Features

- Gallager regular LDPC parity-check matrix construction
- GF(2) rank analysis and actual code-rate diagnostics
- Systematic encoding from generator matrices
- Binary Symmetric Channel (BSC)
- Additive White Gaussian Noise (AWGN) channel with BPSK LLRs
- Binary Erasure Channel (BEC)
- Bit-Flip decoder for BSC hard-decision decoding
- Belief Propagation decoder
- Min-Sum decoder
- Monte Carlo BER and FER simulations
- Convergence-rate and average-iteration tracking
- Signed coding-gain calculation at every channel point
- Validation warnings for suspicious or poor decoder behavior
- PDF report generation from saved figures, tables, and summary JSON

## Project Structure

```text
LDPC-Codes-Implementation/
+-- experiment.py
+-- generate_report.py
+-- ldpc.py
+-- ldpc_construction.py
+-- requirements.txt
+-- report.pdf
+-- results/
|   +-- figures/
|   |   +-- bsc_ber_fer.png
|   |   +-- awgn_ber_fer.png
|   |   +-- bec_ber_fer.png
|   |   +-- decoder_convergence.png
|   |   +-- coding_gain_summary.png
|   +-- tables/
|   |   +-- bsc_results.csv
|   |   +-- awgn_results.csv
|   |   +-- bec_results.csv
|   +-- summaries/
|       +-- experiment_summary.json
+-- tests/
+-- legacy/
    +-- experiments_20_10.py
    +-- simple_demo_20_10.py
```

Some generated files may not exist until `python experiment.py` is run.

## Installation

Clone the repository:

```bash
git clone https://github.com/faisaliqbal946/LDPC-Codes-Implementation.git
cd LDPC-Codes-Implementation
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

Run the main large-code Monte Carlo experiment:

```bash
python experiment.py
```

The default simulation uses:

- target length `200`, resolved to `N = 204`
- actual dimension `K = 104`
- `300` frames per channel point
- `50` BP/Min-Sum iterations
- `15` Bit-Flip iterations

Optional quick smoke run:

```bash
EXPERIMENT_QUICK=1 EXPERIMENT_FRAMES=50 python experiment.py
```

Windows PowerShell equivalent:

```powershell
$env:EXPERIMENT_QUICK="1"
$env:EXPERIMENT_FRAMES="50"
python experiment.py
```

Run tests:

```bash
pytest
```

## Generated Outputs

The experiment writes figures to:

```text
results/figures/
```

Expected figures:

- `bsc_ber_fer.png`
- `awgn_ber_fer.png`
- `bec_ber_fer.png`
- `decoder_convergence.png`
- `coding_gain_summary.png`

The experiment writes numeric CSV tables to:

```text
results/tables/
```

Expected tables:

- `bsc_results.csv`
- `awgn_results.csv`
- `bec_results.csv`

The experiment writes summary diagnostics to:

```text
results/summaries/experiment_summary.json
```

Each CSV row includes uncoded BER, decoder BER, decoder FER, convergence rate, average iterations, and coding gain in dB.

## Report Generation

Generate the PDF report:

```bash
python generate_report.py
```

The generated report is written to:

```text
report.pdf
```

The report focuses on the large `(204,104)` LDPC experiment and reads from:

- `results/figures/`
- `results/tables/`
- `results/summaries/experiment_summary.json`

If expected outputs are missing, rerun `python experiment.py` before generating the report.

## Validation Checks

The experiment records validation warnings in:

```text
results/summaries/experiment_summary.json
```

Warnings are not hidden. They may indicate:

- coded BER worse than uncoded BER at many channel points
- BP worse than Min-Sum at most BSC/AWGN points
- low convergence rate
- exactly identical BP and Min-Sum behavior outside BEC
- many detected 4-cycles in the parity-check matrix

These checks are intended to make the results easier to audit, not to force the simulation to look favorable.

## Known Limitations

- The code length is finite: `(204,104)` is useful for coursework simulation but not a production LDPC length.
- The default Monte Carlo budget is `300` frames per point, so curves may contain sampling noise.
- The Gallager regular construction is not optimized for girth or degree-distribution performance.
- Short cycles may exist and can reduce iterative-decoder performance.
- Negative coding gain may occur at some operating points and is reported honestly.
- The implementation is pure Python and prioritizes clarity over speed.
- Results are coursework-level and should not be described as capacity-achieving.

## References

1. Gallager, R. G., *Low-Density Parity-Check Codes*, MIT Press, 1963.
2. MacKay, D. J. C., *Information Theory, Inference, and Learning Algorithms*, Cambridge University Press, 2003.
3. Richardson, T. and Urbanke, R., *Modern Coding Theory*, Cambridge University Press, 2008.
4. Shannon, C. E., "A Mathematical Theory of Communication," *Bell System Technical Journal*, 1948.

## Archived Educational Files

The current main workflow is the large `(204,104)` simulation. Older small `(20,10)` educational scripts are preserved only for reference:

```text
legacy/experiments_20_10.py
legacy/simple_demo_20_10.py
```

They are not part of the main experiment or report workflow.

## Author

**Faisal Iqbal**  
Course: Information Theory  
Project: LDPC Codes Implementation  
Language: Python
