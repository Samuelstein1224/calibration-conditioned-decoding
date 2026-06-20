# In a new file, e.g., model_final.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool


# --- Conditioning Components (FiLM and GNN Encoder are unchanged) ---
class FiLMLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        return gamma * x + beta


class SystemEncoderGNN(nn.Module):
    def __init__(self, node_features, edge_features, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, data: Batch):
        if data.x is None or data.edge_index is None:
            return torch.zeros(data.num_graphs, self.embedding_dim).to(data.x.device if data.x is not None else 'cpu')
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return global_mean_pool(x, batch)


class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class FiLMedConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.film = FiLMLayer()

    def forward(self, x, gamma, beta):
        x = self.conv(x)
        x = self.film(x, gamma, beta)
        x = self.relu(x)
        return x



class GeneralRepCodeDecoder(nn.Module):
    """
    The definitive decoder model. It is:
    1. GENERAL: Handles variable D and T via padding.
    2. NOT CONDITIONED! DOES NOT TAKE IN THE SYSTEM PROPERTIES!.
    """

    def __init__(self, D_max, r_max, system_node_features, system_edge_features,
                 channels=64, embedding_dim=128):
        super().__init__()
        self.D_max = D_max
        self.r_max = r_max

        # --- Main Decoder Arm (2D CNN with FiLM) ---
        self.conv_block1 = Conv2DBlock(1, channels)
        self.conv_block2 = Conv2DBlock(channels, channels * 2)
        self.conv_block3 = Conv2DBlock(channels * 2, channels * 4)


        num_stabilizers_max = D_max - 1
        self.flattened_size_max = (channels * 4) * self.r_max * num_stabilizers_max
        self.output_head = nn.Linear(self.flattened_size_max, self.D_max)

    def forward(self, syndrome_block, system_properties):
        N, _, r_current, S_current = syndrome_block.shape
        D_current = S_current + 1

        # 1. Get Conditioning Vector from GNN


        # 4. Run the main decoder, injecting FiLM at each block
        x = self.conv_block1(syndrome_block)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # x shape is now (N, channels*4, r_current, S_current)

        # 5. Flatten the feature map
        # Flatten is stride-agnostic (safe for channels_last or non-contiguous tensors)
        x_flattened = torch.flatten(x, start_dim=1)

        # 6. Pad the flattened vector to the maximum expected size
        current_flattened_size = x_flattened.shape[1]
        padding_needed = self.flattened_size_max - current_flattened_size

        if padding_needed < 0:
            raise ValueError(
                f"Input size ({r_current}, {S_current}) exceeds model's max size ({self.r_max}, {self.D_max - 1})")

        x_padded = F.pad(x_flattened, (0, padding_needed))

        logits = self.output_head(x_padded)

        # 8. Reshape and slice the output to match the *current* D
        logits = logits.view(-1, 1, self.D_max)
        final_logits = logits[:, :, :D_current]

        return torch.sigmoid(final_logits)

class GeneralConditionedRepCodeDecoder(nn.Module):
    """
    The definitive decoder model. It is:
    1. GENERAL: Handles variable D and T via padding.
    2. CONDITIONED: Adapts to hardware noise via a GNN and FiLM layers.
    """

    def __init__(self, D_max, r_max, system_node_features, system_edge_features,
                 channels=64, embedding_dim=128):
        super().__init__()
        self.D_max = D_max
        self.r_max = r_max

        # --- Conditioning Arm ---
        self.system_encoder = SystemEncoderGNN(
            node_features=system_node_features,
            edge_features=system_edge_features,
            embedding_dim=embedding_dim
        )

        # --- Main Decoder Arm (2D CNN with FiLM) ---
        self.conv_block1 = FiLMedConv2DBlock(1, channels)
        self.conv_block2 = FiLMedConv2DBlock(channels, channels * 2)
        self.conv_block3 = FiLMedConv2DBlock(channels * 2, channels * 4)

        # --- FiLM Generator ---
        film_params_c1 = channels * 2
        film_params_c2 = (channels * 2) * 2
        film_params_c3 = (channels * 4) * 2
        total_film_params = film_params_c1 + film_params_c2 + film_params_c3

        self.film_generator = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, total_film_params)
        )

        # --- Output Head (Fixed to MAX possible size) ---
        num_stabilizers_max = D_max - 1
        self.flattened_size_max = (channels * 4) * self.r_max * num_stabilizers_max
        self.output_head = nn.Linear(self.flattened_size_max, self.D_max)

    def forward(self, syndrome_block, system_properties):
        # syndrome_block shape: (N, 1, r_current, D_current-1)
        N, _, r_current, S_current = syndrome_block.shape
        D_current = S_current + 1

        # 1. Get Conditioning Vector from GNN
        conditioning_vector = self.system_encoder(system_properties)

        # 2. Generate all FiLM parameters
        all_film_params = self.film_generator(conditioning_vector)

        # 3. Unpack FiLM parameters for each block
        c1, c2, c4 = self.conv_block1.conv.out_channels, self.conv_block2.conv.out_channels, self.conv_block3.conv.out_channels

        params_c1 = all_film_params[:, :c1 * 2]
        params_c2 = all_film_params[:, c1 * 2: c1 * 2 + c2 * 2]
        params_c3 = all_film_params[:, c1 * 2 + c2 * 2:]

        gamma1, beta1 = torch.split(params_c1, c1, dim=1)
        gamma2, beta2 = torch.split(params_c2, c2, dim=1)
        gamma3, beta3 = torch.split(params_c3, c4, dim=1)
        # If you turn this on below it will make the film layers do nothing.
        #
        # gamma1 = torch.ones(N, c1, device=syndrome_block.device)
        # beta1  = torch.zeros(N, c1, device=syndrome_block.device)
        # gamma2 = torch.ones(N, c2, device=syndrome_block.device)
        # beta2  = torch.zeros(N, c2, device=syndrome_block.device)
        # gamma3 = torch.ones(N, c4, device=syndrome_block.device)
        # beta3  = torch.zeros(N, c4, device=syndrome_block.device)

        # 4. Run the main decoder, injecting FiLM at each block
        x = self.conv_block1(syndrome_block, gamma1, beta1)
        x = self.conv_block2(x, gamma2, beta2)
        x = self.conv_block3(x, gamma3, beta3)
        # x shape is now (N, channels*4, r_current, S_current)

        # 5. Flatten the feature map
        # Flatten is stride-agnostic (safe for channels_last or non-contiguous tensors)
        x_flattened = torch.flatten(x, start_dim=1)

        # 6. Pad the flattened vector to the maximum expected size
        current_flattened_size = x_flattened.shape[1]
        padding_needed = self.flattened_size_max - current_flattened_size

        if padding_needed < 0:
            raise ValueError(
                f"Input size ({r_current}, {S_current}) exceeds model's max size ({self.r_max}, {self.D_max - 1})")

        x_padded = F.pad(x_flattened, (0, padding_needed))

        # 7. Pass the fixed-size vector through the output head
        logits = self.output_head(x_padded)

        # 8. Reshape and slice the output to match the *current* D
        logits = logits.view(-1, 1, self.D_max)
        final_logits = logits[:, :, :D_current]

        return torch.sigmoid(final_logits)
