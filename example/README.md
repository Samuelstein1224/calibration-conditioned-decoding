# Worked Example: End-to-End ML Decoder for Repetition Codes

This directory contains a self-contained example that walks through the full
decoder pipeline — from raw experimental data to trained model to evaluation —
using a small sample of real IBM quantum hardware data.

## What's Included

```
example/
├── README.md                  # This file
├── run_example.py             # End-to-end worked example (run this)
├── create_sample_data.py      # Script that generated the sample data
└── sample_data/               # 256 shots from real IBM hardware (~680KB)
    ├── job_z/                 # D=11, r=11, Z-basis measurements
    │   ├── shots.pt           # 256 syndrome shots (128 per logical state)
    │   ├── metadata.pt        # Hardware graph with calibration features
    │   └── params.json        # Job metadata (D, r, basis)
    └── job_x/                 # D=11, r=11, X-basis measurements
        ├── shots.pt
        ├── metadata.pt
        └── params.json
```

## Quick Start

```bash
# From the repo root:
pip install -e .
python example/run_example.py
```

## What the Example Does

**Step 1 — Inspect the data**: Loads the sample data and prints the structure
of syndrome blocks, correction labels, and hardware calibration graphs. This
shows what real IBM quantum hardware data looks like after preprocessing.

**Step 2 — Train a decoder**: Trains a `GeneralRepCodeDecoder` (Conv2D backbone)
on 179 training shots for 30 epochs. Uses smaller channels (32 vs 128) for speed.
Takes ~10 seconds on GPU.

**Step 3 — Evaluate**: Runs `evaluate_decoder()` on the 77-shot validation split.
Reports logical error rates for |0> and |1> states, overall accuracy, pre/post
correction confidence, and binomial confidence intervals.

**Step 4 — Pretrained model** (optional): If `models/r11_D11_Z/` exists
locally, loads the full pretrained FiLM-conditioned model and evaluates it on
the same sample data. Compares with published validation metrics.

## Sample Data Details

The sample data is extracted from real experiments on IBM quantum hardware
(ibm_fez backend). Each shot contains:

- **syndrome_block** `(1, 11, 10)`: 11 rounds of 10 stabilizer measurements
- **target_correction** `(11,)`: ground-truth Pauli correction frame
- **final_measurement** `(11,)`: final data qubit readout
- **intended_logical_state**: 0 or 1

The hardware graph encodes per-qubit calibration data:
- Node features: T1 (normalized), T2 (normalized), readout error, SX gate error
- Edge features: two-qubit gate error (CZ/ECR)
- 21 nodes per chain (11 data qubits + 10 ancilla qubits)

## Adapting This Example

To train on the full dataset:
```bash
python scripts/train.py processed_data/ --filter-D 11 --filter-r 11 --filter-basis Z
```

Select the decoder variant with `--model`:
```bash
python scripts/train.py processed_data/ --filter-D 11 --filter-r 11 --filter-basis Z --model cnn    # unconditioned baseline
python scripts/train.py processed_data/ --filter-D 11 --filter-r 11 --filter-basis Z --model film   # calibration-conditioned FiLM decoder
```
