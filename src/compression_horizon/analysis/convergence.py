import torch


class ConvergenceTracker:
    """Tracks per-sample steps spent below given convergence thresholds."""

    def __init__(
        self,
        max_optimization_steps: int,
        batch_size: int,
        thresholds: tuple[float, ...] = (0.95, 0.99, 1.0),
    ):
        self._buffers = {
            threshold: torch.zeros((max_optimization_steps, batch_size), dtype=torch.long) for threshold in thresholds
        }
        self.fully_converged: torch.Tensor | None = None  # [batch]

    def update(self, step_i: int, convergence_per_sample: torch.Tensor) -> bool:
        """Record one step. Returns True if every batch sample reached convergence == 1.0."""
        for threshold, buffer in self._buffers.items():
            buffer[step_i, :] = (convergence_per_sample < threshold).to(torch.long)
        self.fully_converged = convergence_per_sample == 1.0
        return self.fully_converged.all().item()

    def steps_below(self, threshold: float) -> torch.Tensor:
        """Per-sample number of steps where convergence was strictly below `threshold`."""
        return self._buffers[threshold].sum(dim=0)


class ConvergedSamplesGuard:
    """Zeroes gradients and restores parameter values for samples already at full convergence."""

    def __init__(self, parameters: torch.Tensor):
        self.parameters = parameters  # [batch, compression, hidden]
        self._snapshot: torch.Tensor | None = None

    def before_step(self, converged_mask: torch.Tensor | None) -> None:
        """Zero out grads for frozen indices and snapshot their current values."""
        if converged_mask is None or self.parameters.grad is None:
            self._snapshot = None
        else:
            self.parameters.grad[converged_mask] = 0
            self._snapshot = self.parameters.detach().clone()

    def after_step(self, converged_mask: torch.Tensor | None) -> None:
        """Restore the pre-step values for frozen indices (undoing weight decay / momentum drift)."""
        if converged_mask is None or self._snapshot is None:
            return
        with torch.no_grad():
            self.parameters[converged_mask] = self._snapshot[converged_mask]
