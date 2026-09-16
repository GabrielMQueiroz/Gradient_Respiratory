import torch
import torch.nn as nn
import torch.nn.functional as F

class SGEMDLayer(nn.Module):
    """
    Structure-Gated Eulerian Micro-Displacement (SG-EMD) Operator.
    
    A differentiable PyTorch layer that amplifies microscopic structural changes.
    The hyper-parameters (alpha, tau, k) are learnable weights.
    
    Mathematical formulation:
        M(I, t) = I(t) + alpha * Mask(J) * tanh(F_t(I(t)) / tau)
    """
    
    def __init__(self, alpha: float = 80.0, tau: float = 1.0, k: float = 1.0, epsilon: float = 1e-5):
        """
        Args:
            alpha (float): Gain factor for amplification.
            tau (float): Anti-blooming clamp.
            k (float): Edge sensitivity factor.
            epsilon (float): Numerical stability constant.
        """
        super(SGEMDLayer, self).__init__()
        
        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor(k, dtype=torch.float32))
        self.epsilon = epsilon
        
        # Sobel kernels for spatial gradients
        sobel_x = torch.tensor([[-1., 0., 1.], 
                                [-2., 0., 2.], 
                                [-1., 0., 1.]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1., -2., -1.], 
                                [ 0.,  0.,  0.], 
                                [ 1.,  2.,  1.]], dtype=torch.float32)
        
        # Register as buffers so they are moved to the correct device but not updated by optimizer
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
    def _compute_spatial_gradients(self, x: torch.Tensor):
        """
        Compute spatial gradients using the Sobel operator.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            gx, gy: Spatial gradients in x and y directions.
        """
        B, C, H, W = x.shape
        # Reshape to treat channels as independent batch elements for convolution
        x_reshaped = x.view(B * C, 1, H, W)
        
        gx = F.conv2d(x_reshaped, self.sobel_x, padding=1)
        gy = F.conv2d(x_reshaped, self.sobel_y, padding=1)
        
        gx = gx.view(B, C, H, W)
        gy = gy.view(B, C, H, W)
        
        return gx, gy

    def _compute_coherence_mask(self, gx: torch.Tensor, gy: torch.Tensor):
        """
        Compute the coherence mask based on the structure tensor.
        C_scaled = [ k^4(J_xx - J_yy)^2 + 4 k^4 J_xy^2 ] / [ k^4(J_xx + J_yy)^2 + eps ]
        """
        k4 = self.k ** 4
        
        # Elements of the structure tensor (local neighborhood)
        # We use a 3x3 average pool (multiplied by 9 to approximate sum)
        # to gather the local neighborhood for the structure tensor.
        J_xx = F.avg_pool2d(gx * gx, kernel_size=3, stride=1, padding=1) * 9.0
        J_yy = F.avg_pool2d(gy * gy, kernel_size=3, stride=1, padding=1) * 9.0
        J_xy = F.avg_pool2d(gx * gy, kernel_size=3, stride=1, padding=1) * 9.0
        
        num = k4 * ((J_xx - J_yy)**2 + 4 * J_xy**2)
        den = k4 * (J_xx + J_yy)**2 + self.epsilon
        
        coherence = num / den
        return coherence

    def forward(self, I_t: torch.Tensor, F_t: torch.Tensor):
        """
        Apply the SG-EMD operator.
        
        Args:
            I_t (torch.Tensor): The current frame(s). Shape (B, C, H, W) or (C, H, W).
            F_t (torch.Tensor): The temporally bandpassed frame(s), 
                                e.g. I(t) - I(t-1) or filtered output. Shape (B, C, H, W) or (C, H, W).
                                
        Returns:
            torch.Tensor: The motion-magnified frame.
        """
        is_unbatched = I_t.dim() == 3
        if is_unbatched:
            I_t = I_t.unsqueeze(0)
            F_t = F_t.unsqueeze(0)

        # 1. Compute spatial gradients
        gx, gy = self._compute_spatial_gradients(I_t)
        
        # 2. Compute coherence mask based on scaled structure tensor
        mask_J = self._compute_coherence_mask(gx, gy)
        
        # 3. Apply Eulerian Micro-Displacement
        # M(I, t) = I(t) + alpha * Mask(J) * tanh(F_t(I(t)) / tau)
        magnified = I_t + self.alpha * mask_J * torch.tanh(F_t / self.tau)
        
        if is_unbatched:
            magnified = magnified.squeeze(0)
            
        return magnified
