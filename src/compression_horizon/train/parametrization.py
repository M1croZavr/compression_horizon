import torch


class DirectParametrization:
    """Embedding optimized directly: parameters == embedding."""

    def __init__(self, init_embedding: torch.Tensor, device: torch.device):
        self.embedding = torch.nn.Parameter(init_embedding.data.to(device))
        self._initialization_snapshot = self.embedding.detach().clone().cpu()

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.embedding]

    @property
    def optimizable_tensor(self) -> torch.Tensor:
        return self.embedding

    def materialize(self) -> torch.Tensor:
        return self.embedding

    def initialization_snapshot(self) -> torch.Tensor:
        return self._initialization_snapshot

    def serialize_extras(self) -> list | None:
        pass


class PretrainedPCAParametrization:
    """Embedding parametrized by low-rank PCA coefficients: e = z @ W + mu."""

    def __init__(
        self,
        batch_size: int,
        num_compression_tokens: int,
        hidden_size: int,
        pca_components: torch.Tensor,
        pca_mean: torch.Tensor,
        device: torch.device,
    ):
        flattened = pca_mean.shape[0]
        expected = num_compression_tokens * hidden_size
        if flattened != expected:
            raise ValueError(
                f"PCA dim mismatch: pretrained has {flattened}, expected {expected} "
                f"(num_tokens={num_compression_tokens}, hidden_size={hidden_size})"
            )
        self._components = pca_components.to(device)
        self._mean = pca_mean.to(device)
        self._batch_size = batch_size
        self._num_compression_tokens = num_compression_tokens
        self._hidden_size = hidden_size
        n_components = self._components.shape[0]
        self.coefficients = torch.nn.Parameter(
            torch.randn([batch_size, n_components], dtype=torch.float32, device=device) * 0.1
        )
        self._initialization_snapshot = self.materialize().detach().cpu()

    @property
    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.coefficients]

    @property
    def optimizable_tensor(self) -> torch.Tensor:
        return self.coefficients

    def materialize(self) -> torch.Tensor:
        flat = torch.matmul(self.coefficients, self._components) + self._mean.unsqueeze(0)
        return flat.reshape(self._batch_size, self._num_compression_tokens, self._hidden_size)

    def initialization_snapshot(self) -> torch.Tensor:
        return self._initialization_snapshot

    def serialize_extras(self) -> list:
        return self.coefficients.clone().detach().to(torch.float32).cpu().numpy().tolist()


def build_parametrization(
    *,
    init_method: str,
    batch_size: int,
    num_compression_tokens: int,
    hidden_size: int,
    device: torch.device,
    init_helper,
    pca_components: torch.Tensor | None,
    pca_mean: torch.Tensor | None,
):
    """Create the parametrization owning the optimizable parameters of the compression embedding."""
    if init_method == "pretrained_pca":
        assert pca_components is not None and pca_mean is not None
        return PretrainedPCAParametrization(
            batch_size=batch_size,
            num_compression_tokens=num_compression_tokens,
            hidden_size=hidden_size,
            pca_components=pca_components,
            pca_mean=pca_mean,
            device=device,
        )
    else:
        return DirectParametrization(init_embedding=init_helper(), device=device)
