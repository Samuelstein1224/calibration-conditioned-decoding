# Experiment Data

This directory holds the raw experimental dataset for *Calibration-Conditioned
FiLM Decoders for Low-Latency Decoding of Quantum Error Correction Evaluated on IBM
Repetition-Code Experiments*, collected on the `ibm_kingston`, `ibm_fez`, and
`ibm_pittsburgh` processors. It contains 352 hardware snapshots spanning code distances
`d = 3, 5, 7, 9, 11` and up to 11 syndrome rounds, in both the `X` and `Z` logical bases,
totalling 3,779,584 measurement shots (20,654,080 individual repetition-code samples
across all parallel chains).

| Device | Snapshots | Shots | Repetition-code samples |
|---|---:|---:|---:|
| `ibm_kingston`   | 155 | 1,902,592 | 10,280,960 |
| `ibm_pittsburgh` | 156 | 1,712,128 |  9,273,344 |
| `ibm_fez`        |  41 |   164,864 |  1,099,776 |
| **Total**        | **352** | **3,779,584** | **20,654,080** |

A "shot" is one circuit execution; each shot reads out several repetition-code chains in
parallel (`n_chains`), so the rep-code-sample count is the per-chain total.

Every experiment is stored raw, exactly as returned by the hardware — no ML-processed or
sparsified form is kept. Experiments are grouped by device and code configuration, and
each snapshot carries its own calibration data, so the structure used in the paper is
recoverable by filtering `index.csv`. Snapshots are anonymized: no IBM Runtime job
identifiers are released.

> The snapshot contents are to be released at Zenodo:
> https://doi.org/10.5281/zenodo.20768087.
> Build or refresh this directory with `python scripts/consolidate_data.py`.

## Layout

```
experiment_data/
├── README.md                  # this file
├── index.csv                  # one row per snapshot — the filter table
└── <backend>/                 # ibm_kingston, ibm_fez, ibm_pittsburgh
    └── d<D>_r<R>/             # code distance D, syndrome rounds R
        └── job_<n>/           # one hardware snapshot (anonymized; no job id)
            ├── info.json          # backend, d, rounds, basis, logical_states, shots, n_chains
            ├── calibration.json   # device calibration snapshot (T1, T2, gate/readout errors, coupling map)
            ├── circuit_state0.qasm # transpiled circuit, logical |0>
            ├── circuit_state1.qasm # transpiled circuit, logical |1>
            └── bitstrings.json     # raw per-shot measurement records
```

Each snapshot runs several distance-`D` repetition-code chains in parallel. In
`bitstrings.json`, every chain appears as a `c_data_*` / `c_syndrome_*` register pair whose
name lists the physical qubits used; values are per-shot bit arrays. `n_chains` in
`info.json` / `index.csv` records how many parallel chains the snapshot contains.

## `index.csv` columns

| column            | description                                                  |
|-------------------|--------------------------------------------------------------|
| `path`            | location of the snapshot, e.g. `ibm_kingston/d11_r11/job_3`  |
| `backend`         | `ibm_kingston`, `ibm_fez`, or `ibm_pittsburgh`               |
| `d`               | code distance                                                |
| `rounds`          | number of syndrome rounds                                    |
| `basis`           | logical basis (`X` / `Z`)                                    |
| `logical_states`  | logical input states present (`0;1`)                         |
| `n_chains`        | number of parallel repetition-code chains in the snapshot    |
| `shots`           | shots per circuit                                            |

## Reproducing the paper splits

Filter `index.csv` on `backend`, `(d, rounds)`, and `basis`. For a given configuration,
multiple snapshots are provided, spanning the training-period and the later
unseen-recalibration experiments of Sec. IV-C; the partitioning used in the paper is
described there. Calibration features for conditioning are read from each snapshot's
`calibration.json`.

## Per-shot calibration

`calibration.json` is the device `backend_properties` captured at submission time: `T1`,
`T2`, gate error rates, and readout assignment errors per qubit, plus coupling-map and
timing data. This is the same calibration consumed by both the FiLM decoder (via the
calibration-graph encoder) and the modified MWPM baseline (via its detector-graph edge
weights).
