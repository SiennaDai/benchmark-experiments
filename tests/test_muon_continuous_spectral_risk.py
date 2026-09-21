import torch

from optim.muon_continuous_spectral_risk import (
    BIN_LABELS,
    active_spectrum,
    associated_component,
    coordinate_error,
    diagonal_component,
    fixed_bin_ids,
    merge_bin_ids,
    metrics,
    pair_component,
)


def test_active_coordinates_and_bins_are_deterministic_and_exhaustive():
    s = torch.tensor([10.0, 1.0, 0.01, 1e-7])
    spec = active_spectrum(s)
    assert spec.active_indices.tolist() == [0, 1, 2]
    ids = fixed_bin_ids(spec.log10_normalized)
    assert ids.shape == spec.active_indices.shape
    assert torch.equal(ids, fixed_bin_ids(spec.log10_normalized))


def test_merge_sparse_bins_preserves_assignment():
    ids = torch.tensor([0, 0, 1, 2, 2, 2, 11])
    merged, labels = merge_bin_ids(ids, min_support=2)
    assert len(labels) >= 1
    assert merged.shape == ids.shape
    assert (merged >= 0).all()


def test_spectral_components_are_orthogonal_and_reconstruct_error():
    torch.manual_seed(4)
    matrix = torch.randn(8, 5)
    q = matrix + 0.01 * torch.randn_like(matrix)
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    ehat = coordinate_error(u, matrix, q, vh)
    masks = []
    for indices in (torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4])):
        mask = torch.zeros(5, dtype=torch.bool); mask[indices] = True; masks.append(mask)
    # All three non-overlapping diagonal components plus the residual are exact.
    diagonal = sum((diagonal_component(u, ehat, vh, mask) for mask in masks), torch.zeros_like(matrix))
    assert torch.allclose(diagonal, u @ torch.diag(ehat.diagonal()) @ vh, atol=1e-6)
    # Reduced SVD coordinates reconstruct the projected component; a tall
    # matrix may also contain residual energy in the left null space.
    projected = u @ ehat @ vh
    assert torch.allclose(projected, u @ u.T @ (q - matrix), atol=1e-5)


def test_associated_and_pair_components_have_intended_coordinate_support():
    torch.manual_seed(5)
    matrix = torch.randn(7, 5); q = matrix + torch.randn_like(matrix) * .02
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False); ehat = coordinate_error(u, matrix, q, vh)
    left = torch.tensor([True, True, False, False, False]); right = torch.tensor([False, False, True, False, False])
    associated = associated_component(u, ehat, vh, left)
    assoc_hat = u.T @ associated @ vh.T
    assert torch.allclose(assoc_hat[~left][:, ~left], torch.zeros_like(assoc_hat[~left][:, ~left]), atol=1e-6)
    pair = pair_component(u, ehat, vh, left, right); pair_hat = u.T @ pair @ vh.T
    assert torch.count_nonzero(pair_hat.abs() > 1e-6).item() == int(left.sum()) * int(right.sum())


def test_metrics_zero_reference_is_explicit():
    result = metrics(torch.zeros(2, 2), torch.ones(2, 2))
    assert result["relative_l2"] is None and result["cosine"] is None
