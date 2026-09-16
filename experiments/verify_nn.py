import torch
from sttensor.nn import SGEMDLayer

def test_sgemd_layer():
    print("Testing SGEMDLayer...")
    layer = SGEMDLayer(alpha=80.0, tau=1.0, k=2.0)
    
    # Fake input video frame (B, C, H, W)
    I_t = torch.rand(2, 3, 64, 64)
    # Fake temporally bandpassed frame
    F_t = torch.randn(2, 3, 64, 64) * 0.1
    
    out = layer(I_t, F_t)
    
    assert out.shape == I_t.shape, f"Expected shape {I_t.shape}, got {out.shape}"
    print(f"Output shape: {out.shape}")
    
    # Ensure backprop works
    loss = out.sum()
    loss.backward()
    
    assert layer.alpha.grad is not None, "alpha gradient is None"
    assert layer.tau.grad is not None, "tau gradient is None"
    assert layer.k.grad is not None, "k gradient is None"
    
    print("Backprop successful, gradients computed.")
    print("SGEMDLayer tests passed!")

if __name__ == "__main__":
    test_sgemd_layer()
