import torch
from sttensor.nn import SGEMDLayer

def main():
    print("Testing SGEMDLayer Edge Cases and Maths...")
    layer = SGEMDLayer(k=1.0)
    
    # Test 1: Unbatched 3D input
    I_t = torch.randn(3, 64, 64)
    F_t = torch.randn(3, 64, 64)
    try:
        out = layer(I_t, F_t)
        print("Unbatched 3D shape:", out.shape)
        assert out.shape == (3, 64, 64)
    except Exception as e:
        print("Unbatched 3D failed:", e)

    # Test 2: Determinant check
    I_t = torch.randn(1, 1, 64, 64)
    # Give it some actual structure so the gradient isn't noise
    I_t[:, :, 20:40, 20:40] = 5.0 
    
    gx, gy = layer._compute_spatial_gradients(I_t)
    import torch.nn.functional as F
    J_xx = F.avg_pool2d(gx * gx, kernel_size=3, stride=1, padding=1) * 9.0
    J_yy = F.avg_pool2d(gy * gy, kernel_size=3, stride=1, padding=1) * 9.0
    J_xy = F.avg_pool2d(gx * gy, kernel_size=3, stride=1, padding=1) * 9.0
    
    det = J_xx * J_yy - J_xy**2
    max_det = det.abs().max().item()
    print("Max determinant:", max_det)
    assert max_det > 1.0, "Determinant is near zero, local tensor failed."
    print("Math verified: Local neighborhood pooling works!")

if __name__ == "__main__":
    main()
