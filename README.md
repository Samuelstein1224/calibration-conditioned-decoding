# Machine Learning Based Decoding: Calibration-Conditioned FiLM Decoders for Low-Latency Decoding of Quantum Error Correction Evaluated on IBM Repetition-Code Experiments

A machine-learning decoder for repetition codes on real IBM quantum hardware, using a FiLM-conditioned GNN + Conv2D architecture that adapts to per-device noise characteristics.

## Architecture

The decoder consists of two arms:

- **SystemEncoderGNN**: A 3-layer GCN that encodes hardware calibration data (T1, T2, readout error, gate errors) into a fixed-size embedding vector.
- **Conv2D Backbone**: Three convolutional blocks (1->64->128->256 channels) that process 2D syndrome detection-event blocks.
- **FiLM Conditioning**: The GNN embedding generates per-block gamma/beta parameters that modulate the conv layers via Feature-wise Linear Modulation, allowing the decoder to adapt to different noise profiles without retraining.

### Variable-size handling (mixed `(d, r)` training)

The architecture is not tied to a single code size, so a single model can be trained on a
dataset that mixes different code distances `D` and syndrome-round counts `r`:

- **Dataset** (`MultiJobQECDataset`) scans all job directories and infers `max_D` / `max_r`
  across the whole (possibly mixed) corpus.
- **Collation** (`variable_size_collate_fn`) pads each sample's syndrome block and labels
  to the largest size in the batch.
- **Model** convolves the padded block, pads the flattened features up to the size implied
  by `(max_D, max_r)`, applies a fixed output head, and then **slices the output back to
  each sample's own `D`**. Inputs larger than `(max_D, max_r)` raise an error.

This means mixed-`(d, r)` datasets train without any architectural change. **In this paper we
train one model per `(d, r)`**, so this capability is supported but not exercised by the
published results.

## Installation

```bash
git clone https://github.com/Samuelstein1224/calibration-conditioned-decoding.git
cd calibration-conditioned-decoding
pip install -e .
```

For development tools:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Training a decoder
```bash
# --model cnn  : unconditioned CNN baseline (default)
# --model film : calibration-conditioned FiLM decoder
python scripts/train.py processed_data/ --filter-basis Z --filter-D 5 7 --model film
```

### Training across a grid of (r, D, basis) configurations
```bash
python scripts/train_multi.py processed_data/ --models-root models
```

### Validating trained models
```bash
python scripts/validate.py models/ processed_validation/
```

### Running MWPM baseline comparison
```bash
python -m ibm_qec.baselines.mwpm_eval -d 5 -t 3 -b X
```

## Project Structure

```
calibration-conditioned-decoding/
├── src/ibm_qec/              # Installable Python package
│   ├── model/                # Decoder architecture (FiLM + Conv2D + GNN)
│   ├── data/                 # Data loading and preprocessing
│   ├── evaluation/           # Evaluation metrics (LER, confidence)
│   └── baselines/            # MWPM decoder baselines
│
├── scripts/                  # Entry-point scripts (consolidate, train, validate)
├── experiment_data/          # All raw hardware experiments (see experiment_data/README.md)
├── example/                  # Minimal runnable example with sample data
└── docs/                     # Additional documentation
```

## Data

The experimental dataset — 352 hardware snapshots spanning code distances `d = 3, 5, 7,
9, 11` and up to 11 syndrome rounds on the IBM `ibm_kingston`, `ibm_fez`, and
`ibm_pittsburgh` processors (3,779,584 measurement shots total) — is to be released at
Zenodo:

> **Zenodo (to be released):** https://doi.org/10.5281/zenodo.20768087

All raw hardware experiments live under [`experiment_data/`](experiment_data/README.md) as
a single flat collection of snapshots. Each snapshot retains its own calibration data, so
the per-device and per-`(d, r, basis)` structure used in the paper is fully recoverable by
filtering `experiment_data/index.csv` — no ML-processed tensors are stored. See
[`experiment_data/README.md`](experiment_data/README.md) for the layout and field
definitions, and `scripts/consolidate_data.py` for how the directory is assembled.

Raw snapshots are converted into ML-ready tensors for training/evaluation with
`python -m ibm_qec.data.prepare`.

## Citation

If you use this code, please cite our paper:

**[Calibration-Conditioned FiLM Decoders for Low-Latency Decoding of Quantum Error Correction Evaluated on IBM Repetition-Code Experiments](https://arxiv.org/abs/2601.16123)**

```bibtex
@article{stein2026calibration,
  title={Calibration-Conditioned FiLM Decoders for Low-Latency Decoding of Quantum Error Correction Evaluated on IBM Repetition-Code Experiments},
  author={Stein, Samuel and Kan, Shuwen and Liu, Chenxu and Harkness, Adrian and Garner, Sean and Du, Zefan and Ding, Yufei and Mao, Ying and Li, Ang},
  journal={arXiv preprint arXiv:2601.16123},
  year={2026}
}
```

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of
Science, National Quantum Information Science Research Centers, Quantum Science Center
(QSC). This research was supported by PNNL's Quantum Algorithms and Architecture for
Domain Science (QuAADS) Laboratory Directed Research and Development (LDRD) Initiative.
The Pacific Northwest National Laboratory is operated by Battelle for the U.S. Department
of Energy under Contract DE-AC05-76RL01830. This research used resources of the Oak Ridge
Leadership Computing Facility (OLCF), which is a DOE Office of Science User Facility
supported under Contract DE-AC05-00OR22725. This research used resources of the National
Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of
Science User Facility located at Lawrence Berkeley National Laboratory, operated under
Contract No. DE-AC02-05CH11231.

## License

Apache 2.0. See [LICENSE](LICENSE).
