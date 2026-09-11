import numpy as np
import pytest
from sttensor.eigen import decompose_tensor, coherence, orientation


def test_decompose_tensor_mathematical_properties():
    np.random.seed(42)
    H, W = 25, 25

    # Generate random symmetric positive semi-definite 2x2 tensors
    d = np.random.randn(H, W, 2)
    # Rank-1 outer product
    T = np.einsum("...i,...j->...ij", d, d)
    # Add random positive diagonal to make rank-2
    T[..., 0, 0] += np.random.uniform(0.1, 2.0, (H, W))
    T[..., 1, 1] += np.random.uniform(0.1, 2.0, (H, W))

    eigvals, eigvecs = decompose_tensor(T)

    assert eigvals.shape == (H, W, 2)
    assert eigvecs.shape == (H, W, 2, 2)

    l1 = eigvals[..., 0]
    l2 = eigvals[..., 1]
    v1 = eigvecs[..., :, 0]
    v2 = eigvecs[..., :, 1]

    # 1. Ordering: lambda1 >= lambda2 >= 0
    assert np.all(l1 >= l2 - 1e-12)
    assert np.all(l2 >= 0.0)

    # 2. Orthonormality of eigenvectors
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    assert np.allclose(norm1, 1.0, atol=1e-12)
    assert np.allclose(norm2, 1.0, atol=1e-12)

    dot12 = np.sum(v1 * v2, axis=-1)
    assert np.allclose(dot12, 0.0, atol=1e-12)

    # 3. Eigenvector equation: T @ v1 = l1 * v1, T @ v2 = l2 * v2
    Tv1 = np.einsum("...ij,...j->...i", T, v1)
    l1_v1 = l1[..., None] * v1
    assert np.allclose(Tv1, l1_v1, atol=1e-10)

    Tv2 = np.einsum("...ij,...j->...i", T, v2)
    l2_v2 = l2[..., None] * v2
    assert np.allclose(Tv2, l2_v2, atol=1e-10)

    # 4. Invariants: trace and determinant
    trace_T = T[..., 0, 0] + T[..., 1, 1]
    assert np.allclose(l1 + l2, trace_T, atol=1e-10)

    det_T = T[..., 0, 0] * T[..., 1, 1] - T[..., 0, 1] * T[..., 1, 0]
    assert np.allclose(l1 * l2, det_T, atol=1e-10)


def test_decompose_degenerate_cases():
    # 1. Zero matrix
    T_zero = np.zeros((1, 1, 2, 2))
    evals, evecs = decompose_tensor(T_zero)
    assert np.allclose(evals, 0.0)
    assert np.allclose(np.linalg.norm(evecs[0, 0, :, 0]), 1.0)
    assert np.allclose(np.linalg.norm(evecs[0, 0, :, 1]), 1.0)

    # 2. Isotropic matrix a*I
    T_iso = np.zeros((1, 1, 2, 2))
    T_iso[0, 0] = np.eye(2) * 5.0
    evals, evecs = decompose_tensor(T_iso)
    assert np.allclose(evals[0, 0], [5.0, 5.0])
    # Must still be orthonormal
    assert np.allclose(np.dot(evecs[0, 0, :, 0], evecs[0, 0, :, 1]), 0.0)

    # 3. Pure y gradient: a=0, b=0, c=9
    T_y = np.zeros((1, 1, 2, 2))
    T_y[0, 0, 1, 1] = 9.0
    evals, evecs = decompose_tensor(T_y)
    assert np.allclose(evals[0, 0], [9.0, 0.0])
    # Major eigenvector should align with y-axis: [0, 1]
    v1 = evecs[0, 0, :, 0]
    assert np.allclose(np.abs(v1), [0.0, 1.0])


def test_decompose_rank_one():
    # Outer product D D^T for D = [3, 4]
    D = np.array([3.0, 4.0])
    T = np.outer(D, D)[None, None, :, :]
    evals, evecs = decompose_tensor(T)

    # lambda1 = norm(D)^2 = 25, lambda2 = 0
    assert np.allclose(evals[0, 0, 0], 25.0)
    assert np.allclose(evals[0, 0, 1], 0.0)

    # Eigenvector should be D / ||D|| = [0.6, 0.8]
    v1 = evecs[0, 0, :, 0]
    assert np.allclose(np.abs(v1), [0.6, 0.8])


def test_coherence():
    # Perfect 1D edge: lambda1 > 0, lambda2 = 0 -> coherence = 1.0
    evals_edge = np.array([[[10.0, 0.0]]])
    coh_edge = coherence(evals_edge)
    assert np.isclose(coh_edge[0, 0], 1.0, atol=1e-5)

    # Perfect isotropic texture: lambda1 == lambda2 -> coherence = 0.0
    evals_iso = np.array([[[5.0, 5.0]]])
    coh_iso = coherence(evals_iso)
    assert np.isclose(coh_iso[0, 0], 0.0, atol=1e-5)

    # Zero energy / flat
    evals_zero = np.array([[[0.0, 0.0]]])
    coh_zero = coherence(evals_zero)
    assert np.isclose(coh_zero[0, 0], 0.0, atol=1e-5)


def test_orientation():
    # Major eigenvector along x-axis [1, 0]
    evecs_x = np.zeros((1, 1, 2, 2))
    evecs_x[0, 0, :, 0] = [1.0, 0.0]
    evecs_x[0, 0, :, 1] = [0.0, 1.0]
    ori_x = orientation(evecs_x)
    assert np.isclose(ori_x[0, 0], 0.0)

    # Major eigenvector along y-axis [0, 1]
    evecs_y = np.zeros((1, 1, 2, 2))
    evecs_y[0, 0, :, 0] = [0.0, 1.0]
    evecs_y[0, 0, :, 1] = [-1.0, 0.0]
    ori_y = orientation(evecs_y)
    assert np.isclose(ori_y[0, 0], np.pi / 2)

    # Diagonal [1/sqrt(2), 1/sqrt(2)]
    evecs_diag = np.zeros((1, 1, 2, 2))
    s = 1.0 / np.sqrt(2.0)
    evecs_diag[0, 0, :, 0] = [s, s]
    evecs_diag[0, 0, :, 1] = [-s, s]
    ori_diag = orientation(evecs_diag)
    assert np.isclose(ori_diag[0, 0], np.pi / 4)

    # modulo_pi check
    evecs_neg_x = np.zeros((1, 1, 2, 2))
    evecs_neg_x[0, 0, :, 0] = [-1.0, 0.0]
    ori_mod = orientation(evecs_neg_x, modulo_pi=True)
    assert np.isclose(ori_mod[0, 0], 0.0, atol=1e-10)


def test_invalid_tensor_shape():
    with pytest.raises(ValueError):
        decompose_tensor(np.zeros((10, 10, 3, 3)))

    with pytest.raises(ValueError):
        coherence(np.zeros((10, 10, 3)))

    with pytest.raises(ValueError):
        orientation(np.zeros((10, 10, 2)))


def test_decompose_tensor_large_values_no_overflow():
    # Values around 1e160 would overflow if squared naively (1e320 > float64 max)
    T = np.array([[[[1e160, 0.0], [0.0, 2e160]]]])
    evals, evecs = decompose_tensor(T)
    assert not np.isinf(evals[0, 0, 0])
    assert np.isclose(evals[0, 0, 0], 2e160)
    assert np.isclose(evals[0, 0, 1], 1e160)


def test_decompose_tensor_batch_dimensions():
    # 5D tensor: (2, 3, 4, 2, 2)
    T = np.random.randn(2, 3, 4, 2, 2)
    T = np.einsum("...ij,...kj->...ik", T, T)
    evals, evecs = decompose_tensor(T)
    assert evals.shape == (2, 3, 4, 2)
    assert evecs.shape == (2, 3, 4, 2, 2)
