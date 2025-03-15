import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention


def latent_rope_operator_tensor(latent, positions, alphas):
    """
    Applies the latent RoPE operator to a tensor of latent vectors.
    
    Args:
        latent (torch.Tensor): Tensor of shape (batch, seq_len, latent_dim), where latent_dim = 2*m.
        positions (torch.Tensor): Tensor of shape (batch, seq_len) containing token positions.
        alphas (torch.Tensor): Tensor of shape (m,) containing frequency parameters for each 2D subspace.
        
    Returns:
        torch.Tensor: The rotated latent tensor, same shape as input.
    """
    # latent shape: (B, L, latent_dim), where latent_dim = 2*m.
    batch, seq_len, latent_dim = latent.size()
    m = latent_dim // 2
    # Create a copy to store rotated values
    latent_rot = latent.clone()
    
    # For each 2D block, apply a rotation by angle = alpha_i * position.
    # The rotation is applied elementwise over the (B, L) dimensions.
    for i in range(m):
        # Compute rotation angle for block i: shape (B, L)
        angle = alphas[i] * positions  
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        # Extract the 2D block for all tokens: shape (B, L, 2)
        block = latent[..., 2*i:2*i+2]
        # Separate components
        x = block[..., 0]
        y = block[..., 1]
        # Apply rotation: [cos, -sin; sin, cos] * [x, y]
        x_rot = cos_angle * x - sin_angle * y
        y_rot = sin_angle * x + cos_angle * y
        # Write the rotated block back
        latent_rot[..., 2*i]   = x_rot
        latent_rot[..., 2*i+1] = y_rot
        
    return latent_rot

class CustomAttentionWithDownProjectionAndLatentRoPE(LlamaAttention):
    def __init__(self, config, layer_idx=None, latent_dim=64):
        """
        Args:
            config: Model configuration (should contain hidden_size).
            layer_idx: Index of the current attention layer.
            latent_dim: Dimension of the compressed latent space (must equal 2*m).
        """
        if layer_idx is None:
            layer_idx = 0
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size
        
        # Down-projection layers for queries and keys.
        self.down_proj_q = nn.Linear(hidden_size, latent_dim, bias=False)
        self.down_proj_k = nn.Linear(hidden_size, latent_dim, bias=False)
        
        # Up-projection layers to recover the original dimension.
        self.up_proj_q = nn.Linear(latent_dim, hidden_size, bias=False)
        self.up_proj_k = nn.Linear(latent_dim, hidden_size, bias=False)
        
        # latent_dim should be even (2*m). Define m.
        assert latent_dim % 2 == 0, "latent_dim must be even."
        m = latent_dim // 2
        
        # Frequency parameters for the RoPE rotations.
        # They can be learnable or fixed. Here we initialize them as learnable parameters.
        self.alphas = nn.Parameter(torch.ones(m))
    
    def forward(self, hidden_states, positions=None, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        """
        Forward pass integrating down-projection, latent RoPE, and up-projection.
        
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (B, L, hidden_size).
            positions (torch.Tensor): Tensor of shape (B, L) with token positions.
            attention_mask, head_mask, output_attentions: Standard attention arguments.
            
        Returns:
            The output of the attention layer (for demonstration, we call the superclass forward).
        """
        # If positions are not provided, generate them assuming sequential tokens.
        if positions is None:
            B, L, _ = hidden_states.size()
            positions = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, L)
        
        # Step 1: Down-project hidden states to latent space.
        latent_q = self.down_proj_q(hidden_states)  # (B, L, latent_dim)
        latent_k = self.down_proj_k(hidden_states)  # (B, L, latent_dim)
        
        # Debug: Print latent shapes.
        print(f"[Layer {self.layer_idx}] Latent Q shape: {latent_q.shape}")
        print(f"[Layer {self.layer_idx}] Latent K shape: {latent_k.shape}")
        
        # Step 2: Apply latent RoPE operator to inject positional information.
        latent_q_rot = latent_rope_operator_tensor(latent_q, positions, self.alphas)
        latent_k_rot = latent_rope_operator_tensor(latent_k, positions, self.alphas)
        
        # Debug: Confirm shapes after RoPE.
        print(f"[Layer {self.layer_idx}] Latent Q (after RoPE) shape: {latent_q_rot.shape}")
        print(f"[Layer {self.layer_idx}] Latent K (after RoPE) shape: {latent_k_rot.shape}")
        
        # Step 3: Recover original dimension via up-projection.
        q = self.up_proj_q(latent_q_rot)
        k = self.up_proj_k(latent_k_rot)
        
        # At this point, you would normally proceed to compute attention using q and k.
        # For demonstration, we call the parent's forward method (which you may modify
        # to accept q and k instead of recomputing them from hidden_states).
        print("Using CustomAttentionWithDownProjectionAndLatentRoPE - proceeding with modified attention computation.")
        
        return super().forward(hidden_states,
                               attention_mask=attention_mask,
                               head_mask=head_mask,
                               output_attentions=output_attentions,
                               **kwargs)
