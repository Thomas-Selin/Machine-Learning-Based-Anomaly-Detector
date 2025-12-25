import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding module for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Initialize positional encoding.

        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length
        """
        super().__init__()
        # Create encoding matrix of zeros [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # Position vector [0,1,2...max_len-1] with shape [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Division terms for sine/cosine wavelengths
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        # Populate matrix with alternating sine/cosine waves
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sine waves
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cosine waves

        # Batch-first storage: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (persistent state not for optimization)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings.

        Expects batch-first tensors: x shape [batch, seq_len, d_model]

        Args:
            x: Input tensor

        Returns:
            Tensor with positional encodings added
        """
        # self.pe: [1, max_len, d_model]
        return x + self.pe[:, : x.size(1), :]


class HybridAutoencoderTransformerModel(nn.Module):
    """Transformer-based autoencoder for microservice anomaly detection

    This model uses:
    - Service-specific embeddings to distinguish between different microservices
    - Cross-service attention to capture dependencies between services
    - Temporal transformer encoder to learn time-series patterns
    - Autoencoder architecture for anomaly detection via reconstruction error
    """

    def __init__(self, num_services: int, num_features: int) -> None:
        """Initialize hybrid autoencoder transformer model.

        Args:
            num_services: Number of microservices
            num_features: Number of features per service
        """
        super().__init__()
        self.hidden_dim = 64
        self.num_services = num_services

        # Service-specific embeddings
        # These embeddings are learnable parameters that are updated during training
        # Each service gets its own unique vector representation (embedding) of size hidden_dim
        # The embeddings are initialized randomly and optimized via backpropagation
        # to capture service-specific characteristics and relationships
        self.service_embeddings = nn.Parameter(
            torch.randn(num_services, self.hidden_dim)
        )

        # Feature processing
        # This linear layer transforms raw features + time features into a hidden representation
        # The +4 accounts for the time features (hour, minute, day, second)
        self.feature_encoder = nn.Linear(num_features + 4, self.hidden_dim)

        # Cross-service attention
        # This allows the model to learn relationships between different services
        self.cross_service_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=4, batch_first=True
        )

        # The expanded dimension for transformer
        transformer_dim = num_services * self.hidden_dim

        # Positional encoding for batch-first transformer input
        self.positional_encoding = PositionalEncoding(transformer_dim)

        # Regular transformer for temporal patterns with matching dimension
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_dim,  # Match the expanded dimension
                nhead=8,  # Make sure nhead divides transformer_dim evenly
                batch_first=True,
            ),
            num_layers=5,
        )

        # Decoder
        self.decoder = nn.Linear(self.hidden_dim, num_features)

    def forward(
        self, x: torch.Tensor, time_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the transformer-based autoencoder

        Args:
            x: Input tensor of shape [batch_size, seq_len, num_services, num_features]
               Contains the raw feature data for each service at each timestep
            time_features: Tensor of shape [batch_size, seq_len, 4]
                          Contains temporal features (hour, minute, day, second)

        Returns:
            Tuple of (reconstructed output, attention_weights)
            Reconstructed output tensor of shape [batch_size, seq_len, num_services, num_features]
            The model attempts to reconstruct the input; reconstruction error indicates anomalies

        Process:
            1. Encode each service's features with service-specific embeddings
            2. Apply cross-service attention to capture inter-service dependencies
            3. Apply temporal transformer to learn time-series patterns
            4. Decode back to original feature space for reconstruction
        """
        batch_size, seq_len, num_services, num_features = x.shape

        # Move time_features to the same device as x
        time_features = time_features.to(x.device)

        # Process each service with its embedding
        service_outputs = []
        for service_idx in range(num_services):
            # Get service data
            service_data = x[:, :, service_idx, :]  # [batch, seq, features]

            # Concatenate service data with time features
            service_input = torch.cat(
                [service_data, time_features], dim=2
            )  # [batch, seq, features+4]

            # Encode features
            encoded = self.feature_encoder(service_input)  # [batch, seq, hidden]

            # Add service embedding
            # This adds the learned service-specific representation to the encoded features
            # The service embedding is the same for all time steps of the same service
            # This helps the model distinguish between different services and learn their unique patterns
            service_emb = self.service_embeddings[service_idx].unsqueeze(0).unsqueeze(0)
            service_emb = service_emb.expand(batch_size, seq_len, -1)
            encoded = encoded + service_emb

            service_outputs.append(encoded)

        # Stack all service representations
        stacked = torch.stack(service_outputs, dim=2)  # [batch, seq, services, hidden]

        # Vectorized cross-service attention across all timesteps
        # Reshape to apply attention across services for all timesteps at once
        stacked_reshaped = stacked.reshape(
            batch_size * seq_len, num_services, self.hidden_dim
        )
        attn_out, _ = self.cross_service_attention(
            stacked_reshaped, stacked_reshaped, stacked_reshaped
        )
        stacked = attn_out.reshape(batch_size, seq_len, num_services, self.hidden_dim)

        # Reshape for transformer (use contiguous() to ensure memory layout compatibility)
        transformer_input = stacked.contiguous().view(
            batch_size, seq_len, -1
        )  # [batch, seq, services*hidden]

        # Add positional encoding
        transformer_input = self.positional_encoding(transformer_input)

        # Pass through transformer
        transformer_out = self.transformer_encoder(transformer_input)

        # Reshape and decode
        transformer_out = transformer_out.view(
            batch_size, seq_len, num_services, self.hidden_dim
        )
        output = self.decoder(transformer_out)

        return output
