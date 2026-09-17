import torch
import torch.nn as nn
import torch.nn.functional as F


class FADN(nn.Module):
    """
    Feature-Attention Depth Network (FADN)

    Input:
        5 surface features:
        SST, SSS, SSH, wind_u, wind_v

    Output:
        Temperature at 6 depths:
        0m, 10m, 30m, 50m, 100m, 200m
    """

    def __init__(self, input_size=5, hidden_size=32, output_size=6):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Learned feature attention
        self.attention_layer = nn.Linear(input_size, input_size)

        # Depth-conditioned prediction head
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # Calculate feature attention weights
        attention_scores = self.attention_layer(x)

        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention to the original input features
        x_attended = x * attention_weights

        # Encode attended features
        encoded = self.encoder(x_attended)

        # Predict temperatures
        temperature = self.depth_head(encoded)

        return temperature, attention_weights