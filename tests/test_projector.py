import torch

from projector.memory import MemoryProjector


def test_projector_is_symmetric_and_idempotent():
    projector = MemoryProjector(embed_dim=3, device=torch.device("cpu"))
    image = torch.tensor([[2.0, 3.0, 0.0, 0.0], [2.0, 0.0, 1.0, 0.0]])
    text = torch.tensor([[2.0, 2.0, 0.0, 0.0], [2.0, 0.0, 2.0, 0.0]])
    projector.update_covariance(image, text)
    matrix = projector.calculate_projector(projector.get_covariance(), 0.8)

    assert torch.allclose(matrix, matrix.T, atol=1e-6)
    assert torch.allclose(matrix @ matrix, matrix, atol=1e-5)


def test_gradient_projection_freezes_time_and_removes_memory_span():
    manager = MemoryProjector(embed_dim=3, device=torch.device("cpu"))
    memory_span = torch.diag(torch.tensor([1.0, 0.0, 0.0]))
    gradient = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    projected = manager.project_gradient(gradient, memory_span)

    assert torch.count_nonzero(projected[:, 0]) == 0
    assert torch.allclose(projected[:, 1:] @ memory_span, torch.zeros(4, 3))
