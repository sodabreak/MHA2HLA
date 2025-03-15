import torch
import torch.nn as nn

class LearnableUpperProjection(nn.Module):
    def __init__(self, W_U: torch.Tensor):
        """
        Initializes the learnable upper-projection module.
        
        Args:
            W_U (torch.Tensor): The original upper projection matrix of shape (d_out, d_in),
                                where d_in = 2*m.
        """
        super().__init__()
        d_out, d_in = W_U.shape
        if d_in % 2 != 0:
            raise ValueError("d_in (latent_dim) must be even.")
        self.d_out = d_out
        self.d_in = d_in
        self.m = d_in // 2  # number of 2D blocks
        
        # Perform SVD on W_U: W_U = U S V^T.
        # U: (d_out, d_out), S: (min(d_out, d_in),), Vh: (d_in, d_in)
        U, S, Vh = torch.linalg.svd(W_U, full_matrices=False)
        
        # Extract B and C:
        # Let B be V from SVD, shape (d_in, d_in), but we only need full latent space.
        B = Vh.T.clone()  # shape: (d_in, d_in)
        # Let C be the first d_in columns of U: shape (d_out, d_in)
        C = U[:, :d_in].clone()
        
        # Register B and C as buffers 
        self.register_buffer("B", B)
        self.register_buffer("C", C)
        
        # Compute optimal initial lambda for each 2D block.
        lambda_init = []
        for i in range(self.m):
            # Use 0-indexing: pair singular values S[2*i] and S[2*i+1]
            # Note: S has length d_in; we assume d_in <= len(S).
            lambda_i = (S[2 * i] + S[2 * i + 1]) / 2.0
            lambda_init.append(lambda_i)
        lambda_init = torch.stack(lambda_init)  # shape: (m,)
        
        # Make lambda parameters learnable.
        self.lambdas = nn.Parameter(lambda_init)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the learnable upper-projection to the latent representation x.
        
        Args:
            x (torch.Tensor): Latent representation of shape (batch, d_in), where d_in = 2*m.
            
        Returns:
            torch.Tensor: Projected representation of shape (batch, d_out).
        """
        # Compute W = sum_{i=1}^{m} C_i * lambda_i * B_i^T.
        # We'll build W from its 2D blocks.
        W = torch.zeros(self.d_out, self.d_in, device=x.device, dtype=x.dtype)
        for i in range(self.m):
            # Extract the i-th 2D block from C and B.
            # C_i: (d_out, 2)
            C_i = self.C[:, 2*i:2*i+2]
            # B_i: (d_in, 2)
            B_i = self.B[:, 2*i:2*i+2]
            # lambda_i is a scalar.
            lambda_i = self.lambdas[i]
            # Contribution: (d_out, 2) @ (2, d_in) => (d_out, d_in)
            W_i = C_i @ (lambda_i * B_i.T)
            W = W + W_i
        
        # Apply the projection: for input x of shape (batch, d_in),
        # we want output = x @ W^T, which results in (batch, d_out)
        output = x @ W.T
        return output

# --------------------- Testing the LearnableUpperProjection module ---------------------

if __name__ == "__main__":
    # Define dimensions.
    d_out = 128
    m = 16              # Number of 2D blocks.
    d_in = 2 * m        # latent_dim
    
    # Create a random original upper-projection matrix W_U 
    W_U = torch.randn(d_out, d_in)
    
    # Initialize the learnable upper-projection module.
    upper_proj_module = LearnableUpperProjection(W_U)
    
    print("Initial lambdas:")
    print(upper_proj_module.lambdas.data)
    
    # Create a dummy latent input (e.g., output of a down-projection) of shape (batch, d_in).
    batch_size = 4
    latent_input = torch.randn(batch_size, d_in)
    
    # Apply the learnable upper projection.
    projected_output = upper_proj_module(latent_input)
    print("\nProjected output shape:", projected_output.shape)
    # Expected shape: (batch_size, d_out)
