import torch
import torch.nn as nn

class DenoiseTransformer(nn.Module):
    """
    A transformer-based neural network that predicts the noise added to text embeddings during the diffusion process.

    This model takes in noised word embeddings along with the current diffusion timestep,
    and learns to predict the noise that was added so it can be subtracted out during reverse diffusion.

    Components:
    - Embedding layer: maps token IDs to vector embeddings
    - Positional encoding: gives the model a sense of word order
    - Timestep embedding: encodes how far along in the noise schedule we are
    - Transformer encoder: models the interactions between tokens in the sequence
    - Output layer: predicts the original noise vector for each token

    Inputs:
    - x: Tensor of shape (B, T, E) → batch of noised embeddings
    - t: Tensor of shape (B,) → timestep values for each example

    Output:
    - A tensor of shape (B, T, E), where each vector is the predicted noise for the corresponding token.
    """
    
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1, max_seq_len=32):
        super().__init__()

        # Learnable embedding layer for token IDs
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Learnable positional embeddings (shape: [1, max_seq_len, embed_dim])
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # A single transformer encoder layer, repeated num_layers times
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True  # Use (B, T, E) instead of (T, B, E)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Embedding for the diffusion timestep (0–999)
        self.time_embed = nn.Embedding(1000, embed_dim)

        # Final layer to project back to original embedding space (for noise prediction)
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, t):
        """
        Forward pass through the denoising transformer.

        Parameters:
        - x: Noised token embeddings of shape (B, T, E)
        - t: Timesteps of shape (B,)

        Returns:
        - Predicted noise for each token embedding: shape (B, T, E)
        """
        B, T, E = x.shape

        # Step 1: Add positional embeddings
        x = x + self.pos_embed[:, :T, :]  # Broadcast to match batch

        # Step 2: Add timestep embeddings (broadcasted to every token)
        t_embed = self.time_embed(t)          # shape: (B, E)
        t_embed = t_embed.unsqueeze(1)        # shape: (B, 1, E)
        t_embed = t_embed.repeat(1, T, 1)      # shape: (B, T, E)
        x = x + t_embed

        # Step 3: Pass through transformer
        x = self.transformer(x)

        # Step 4: Predict the noise
        return self.output_layer(x)
