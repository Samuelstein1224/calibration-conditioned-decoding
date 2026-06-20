# Model Architecture

## Overview

The decoder uses a two-arm architecture: a GNN-based system encoder produces a hardware-aware embedding, which conditions a Conv2D syndrome decoder via FiLM (Feature-wise Linear Modulation).

## SystemEncoderGNN

Encodes the physical hardware graph into a fixed-size vector.

- **Input**: PyTorch Geometric `Data` object with:
  - Node features (per qubit): T1 (normalized), T2 (normalized), readout error, SX gate error
  - Edge features: CZ/ECR gate error
  - Edge index: hardware coupling map
- **Architecture**: 3-layer GCN (`GCNConv`) with ReLU activations
  - Layer 1: `node_features` -> `hidden_dim` (256)
  - Layer 2: `hidden_dim` -> `hidden_dim` (256)
  - Layer 3: `hidden_dim` -> `embedding_dim` (128)
- **Pooling**: `global_mean_pool` over all nodes -> single vector per graph
- **Output**: `(batch_size, embedding_dim)` tensor

## Conv2D Backbone

Processes the 2D syndrome detection-event block.

- **Input**: `(N, 1, r, D-1)` tensor of detection events
  - r = number of syndrome rounds
  - D-1 = number of stabilizers
- **Architecture**: 3 convolutional blocks
  - Block 1: `Conv2d(1, channels, 3, padding=1)` + ReLU
  - Block 2: `Conv2d(channels, channels*2, 3, padding=1)` + ReLU
  - Block 3: `Conv2d(channels*2, channels*4, 3, padding=1)` + ReLU
- Default `channels=64`, giving 64->128->256 channel progression
- **Output head**: Flatten -> pad to max size -> `Linear(flattened_max, D_max * 2)` -> reshape to `(N, 2, D)` -> sigmoid

## FiLM Conditioning

The GNN embedding drives a FiLM generator that produces per-block affine parameters:

```
embedding -> Linear(embedding_dim, 256) -> ReLU -> Linear(256, total_film_params)
```

For each conv block `i`, the FiLM layer applies:
```
output_i = gamma_i * conv_i(x) + beta_i
```

where `gamma_i` and `beta_i` are `(N, C_i)` vectors broadcast across spatial dimensions.

Total FiLM parameters: `2*C1 + 2*C2 + 2*C3` = `2*64 + 2*128 + 2*256` = 896 (at default settings).

## Variable Size Handling

The model supports variable D and r at inference time:
1. Syndrome blocks are padded to `(r_max, D_max-1)` within each batch
2. The flattened feature vector is zero-padded to the maximum expected size
3. The output is sliced to `[:, :, :D_current]` to match the actual code distance

## Model Variants

- **GeneralConditionedRepCodeDecoder**: Full FiLM-conditioned model (GNN + Conv2D + FiLM)
- **GeneralRepCodeDecoder**: Conv2D-only model without FiLM conditioning (ablation baseline)

Both share the same Conv2D backbone and output head architecture.
