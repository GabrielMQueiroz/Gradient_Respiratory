import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_structure_tensor_lambda1(image):
    # Sobel Gx, Gy
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Structure tensor components
    Ixx = gx ** 2
    Iyy = gy ** 2
    Ixy = gx * gy
    
    # Smooth the components (optional but standard for structure tensor, typically small window)
    Ixx = cv2.GaussianBlur(Ixx, (5, 5), 1.0)
    Iyy = cv2.GaussianBlur(Iyy, (5, 5), 1.0)
    Ixy = cv2.GaussianBlur(Ixy, (5, 5), 1.0)
    
    # Eigenvalues: lambda1 = 0.5 * (Ixx + Iyy + sqrt((Ixx - Iyy)^2 + 4*Ixy^2))
    trace = Ixx + Iyy
    det_term = np.sqrt((Ixx - Iyy)**2 + 4 * Ixy**2)
    lambda1 = 0.5 * (trace + det_term)
    
    return np.mean(lambda1)

def generate_base_texture(size=256, frequency=10):
    # Checkerboard-like pattern
    x = np.linspace(0, 2*np.pi*frequency, size)
    y = np.linspace(0, 2*np.pi*frequency, size)
    X, Y = np.meshgrid(x, y)
    texture = np.sin(X) * np.sin(Y)
    # Normalize to 0-255
    texture = ((texture + 1) / 2 * 255).astype(np.uint8)
    return texture

def test_geometric_dilation():
    size = 256
    base_tex = generate_base_texture(size, frequency=15)
    
    t_vals = np.linspace(0, 10, 100)
    alpha_vals = 1 + 0.05 * np.sin(2 * np.pi * 0.2 * t_vals)
    
    lambda1_vals = []
    
    for alpha in alpha_vals:
        # Resize according to alpha (geometric dilation)
        new_size = int(size * alpha)
        if new_size < size:
            # Crop to original size
            resized = cv2.resize(base_tex, (new_size, new_size))
            pad_before = (size - new_size) // 2
            pad_after = size - new_size - pad_before
            img = np.pad(resized, ((pad_before, pad_after), (pad_before, pad_after)), mode='reflect')
        else:
            # Crop center to original size
            resized = cv2.resize(base_tex, (new_size, new_size))
            crop_start = (new_size - size) // 2
            img = resized[crop_start:crop_start+size, crop_start:crop_start+size]
        
        l1 = compute_structure_tensor_lambda1(img)
        lambda1_vals.append(l1)
        
    return t_vals, alpha_vals, np.array(lambda1_vals)

def test_wrinkle_density():
    size = 256
    t_vals = np.linspace(0, 10, 100)
    density_vals = 15 + 2 * np.sin(2 * np.pi * 0.2 * t_vals)
    
    lambda1_vals = []
    
    for d in density_vals:
        # Re-generate texture with different frequency (wrinkle density)
        img = generate_base_texture(size, frequency=d)
        l1 = compute_structure_tensor_lambda1(img)
        lambda1_vals.append(l1)
        
    return t_vals, density_vals, np.array(lambda1_vals)

if __name__ == "__main__":
    t, alpha, l1_dilation = test_geometric_dilation()
    
    # Correlation between alpha and lambda1 for dilation
    corr_dilation = np.corrcoef(alpha, l1_dilation)[0, 1]
    print(f"Geometric Dilation Correlation (alpha vs lambda1): {corr_dilation:.3f}")
    
    t2, density, l1_density = test_wrinkle_density()
    corr_density = np.corrcoef(density, l1_density)[0, 1]
    print(f"Wrinkle Density Correlation (density vs lambda1): {corr_density:.3f}")
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(t, alpha, label='alpha(t) - Stretch')
    axs[0].plot(t, l1_dilation / np.max(l1_dilation), label='Normalized lambda1')
    axs[0].set_title('Pure Geometric Dilation')
    axs[0].legend()
    
    axs[1].plot(t2, density / np.max(density), label='Density(t) - Normalized')
    axs[1].plot(t2, l1_density / np.max(l1_density), label='Normalized lambda1')
    axs[1].set_title('Pure Wrinkle Density Modulation')
    axs[1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_validation.png')
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
