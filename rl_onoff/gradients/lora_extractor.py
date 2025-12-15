"""
LoRA gradient extraction and projection utilities.

- LoraBGradientProjector: computes per-token gradients w.r.t. LoRA B matrices
  and projects them to a lower-dimensional space using a Projector
  (BasicProjector or CudaProjector).
"""

from typing import List, Dict, Tuple, Union

import numpy as np
import torch

from rl_onoff.backends.base import BaseBackend
from rl_onoff.gradients.projectors import (
    BasicProjector,
    CudaProjector,
    ProjectionType,
)


class LoraBGradientProjector:
    """
    Compute per-token gradients w.r.t. LoRA B matrices and project them
    to a lower-dimensional space using a random projector.

    For each sample:
      - raw token gradients: shape (k, grad_dim)
      - projected gradients: shape (k, proj_dim)
    """

    def __init__(
        self,
        backend: BaseBackend,
        proj_dim: int,
        device: Union[str, torch.device] = "cuda",
        proj_type: str = "rademacher",
        use_cuda_projector: bool = False,
        block_size: int = 100,
        seed: int = 0,
        cuda_max_batch_size: int = 32,
    ) -> None:
        """
        Args:
            backend: HuggingFaceBackend instance with LoRA attached.
            proj_dim: target projection dimension d.
            device: device for projector ("cuda" recommended).
            proj_type: "normal" or "rademacher".
            use_cuda_projector: if True, use CudaProjector; else BasicProjector.
            block_size: block size for BasicProjector (if used).
            seed: random seed for projector.
            cuda_max_batch_size: max batch size for CudaProjector.
        """
        self.backend = backend
        self.device = torch.device(device)
        self.proj_dim = proj_dim
        self.proj_type = ProjectionType(proj_type)
        self.seed = seed

        # Infer grad_dim (total number of LoRA-B parameters) and layout
        self.lora_b_param_names, self.param_shapes, self.grad_dim = (
            self._inspect_lora_b_parameters()
        )

        # Initialize projector
        if use_cuda_projector:
            self.projector = CudaProjector(
                grad_dim=self.grad_dim,
                proj_dim=self.proj_dim,
                seed=self.seed,
                proj_type=self.proj_type,
                device=self.device,
                max_batch_size=cuda_max_batch_size,
            )
        else:
            self.projector = BasicProjector(
                grad_dim=self.grad_dim,
                proj_dim=self.proj_dim,
                seed=self.seed,
                proj_type=self.proj_type,
                device=self.device,
                block_size=block_size,
            )

    def _inspect_lora_b_parameters(
        self,
    ) -> Tuple[List[str], Dict[str, Tuple[int, ...]], int]:
        """
        Find all LoRA B parameters and compute the total gradient dimension.
        Assumes LoRA B parameters have 'lora_B' in their name.
        """
        if not hasattr(self.backend, "model") or self.backend.model is None:
            raise RuntimeError("Backend model not loaded. Call backend.load() first.")

        names: List[str] = []
        shapes: Dict[str, Tuple[int, ...]] = {}
        grad_dim = 0

        for name, param in self.backend.model.named_parameters():
            if "lora" in name.lower() and "lora_b" in name.lower():
                names.append(name)
                shapes[name] = tuple(param.shape)
                grad_dim += param.numel()

        if grad_dim == 0:
            raise ValueError("No LoRA B parameters found (names containing 'lora_B').")

        return names, shapes, grad_dim

    def _vectorize_lora_b_gradients(
        self,
        grad_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Convert a dict of LoRA-B gradients into a (k, grad_dim) matrix.

        grad_dict: {param_name: np.ndarray (k, *param_shape)}
        Returns:
            token_grads: np.ndarray (k, grad_dim)
        """
        k = None
        flat_grads_per_param: List[np.ndarray] = []

        for name in self.lora_b_param_names:
            if name not in grad_dict:
                # If missing, treat as zeros
                shape = self.param_shapes[name]
                if k is None:
                    # infer k from any present param
                    for other in grad_dict.values():
                        k = other.shape[0]
                        break
                    if k is None:
                        raise ValueError("Empty gradient dict; cannot infer num_tokens.")
                flat_grads_per_param.append(
                    np.zeros((k, int(np.prod(shape))), dtype=np.float32)
                )
                continue

            g = grad_dict[name]  # (k, *param_shape)
            if k is None:
                k = g.shape[0]
            assert g.shape[0] == k, "All LoRA-B grads must have same num_tokens."

            flat = g.reshape(k, -1)  # (k, param_numel)
            flat_grads_per_param.append(flat)

        token_grads = np.concatenate(flat_grads_per_param, axis=1)  # (k, grad_dim)
        return token_grads

    def compute_token_gradients(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
    ) -> List[np.ndarray]:
        """
        Compute raw per-token LoRA-B gradients (unprojected).

        Returns:
            List of np.ndarray, one per sample, each with shape (k, grad_dim).
        """
        raw = self.backend.get_lora_gradients(
            prompts=prompts,
            responses=responses,
            reduction="none",
        )
        # raw: dict (single) or list[dict]

        token_grad_mats: List[np.ndarray] = []

        if isinstance(raw, dict):
            token_grad_mats.append(self._vectorize_lora_b_gradients(raw))
        else:
            for grad_dict in raw:
                token_grad_mats.append(self._vectorize_lora_b_gradients(grad_dict))

        return token_grad_mats

    def project_token_gradients(
        self,
        token_grads: np.ndarray,
        model_id: int = 0,
    ) -> np.ndarray:
        """
        Project per-token gradients to lower dimension.

        Args:
            token_grads: np.ndarray (k, grad_dim)
            model_id: projector model ID (to change projection matrix if desired).

        Returns:
            projected: np.ndarray (k, proj_dim)
        """
        grads_tensor = torch.from_numpy(token_grads).to(self.device)  # (k, grad_dim)
        with torch.no_grad():
            projected = self.projector.project(grads_tensor, model_id=model_id)
        return projected.cpu().numpy()

    def compute_and_project(
        self,
        prompts: Union[str, List[str]],
        responses: Union[str, List[str]],
        model_id: int = 0,
    ) -> List[np.ndarray]:
        """
        Convenience: compute LoRA-B gradients and project them.

        Returns:
            List of np.ndarray, one per sample, each with shape (k, proj_dim).
        """
        token_grads_list = self.compute_token_gradients(prompts, responses)
        projected_list: List[np.ndarray] = []

        for i, tg in enumerate(token_grads_list):
            projected = self.project_token_gradients(
                tg,
                model_id=model_id if len(token_grads_list) == 1 else model_id + i,
            )
            projected_list.append(projected)

        return projected_list


if __name__ == "__main__":
    """Simple smoke test for LoraBGradientProjector with a dummy backend."""

    import torch.nn as nn
    from rl_onoff.backends.base import BaseBackend

    class DummyModel(nn.Module):
        """Tiny model with a single LoRA B parameter for testing."""

        def __init__(self) -> None:
            super().__init__()
            # Simulate a LoRA B matrix (e.g., out_dim x r)
            self.test_lora_B = nn.Parameter(torch.randn(8, 2))

    class DummyBackend(BaseBackend):
        """Backend stub that exposes LoRA-B gradients in the expected format."""

        def __init__(self) -> None:
            super().__init__("dummy-model")
            self.model = DummyModel()

        def load(self) -> None:  # noqa: D401
            """No-op load for dummy backend."""
            return

        def generate(self, *args, **kwargs):
            raise NotImplementedError

        def get_logits(self, *args, **kwargs):
            raise NotImplementedError
        
        def get_tokenizer(self):
            """Dummy tokenizer accessor for testing."""
            return None

        def get_lora_gradients(
            self,
            prompts: Union[str, List[str]],
            responses: Union[str, List[str]],
            reduction: str = "none",
        ):
            # For testing, we ignore the actual prompts/responses and just
            # return synthetic gradients matching the LoRA-B parameter shapes.
            if isinstance(prompts, str):
                batch_size = 1
            else:
                batch_size = len(prompts)

            # Simulate k tokens per sample
            num_tokens = 5

            # Collect LoRA-B params
            lora_b_params = {
                name: p
                for name, p in self.model.named_parameters()
                if "lora_b" in name.lower() or "lora_B" in name
            }
            # If our simple model doesn't follow that naming, just use the test param
            if not lora_b_params:
                lora_b_params = {
                    "test_lora_B": self.model.test_lora_B,
                }

            def make_grad_dict():
                out: Dict[str, np.ndarray] = {}
                for name, p in lora_b_params.items():
                    shape = p.shape  # e.g., (8, 2)
                    g = np.random.randn(num_tokens, *shape).astype(np.float32)
                    out[name] = g
                return out

            if reduction == "none":
                if batch_size == 1:
                    return make_grad_dict()
                else:
                    return [make_grad_dict() for _ in range(batch_size)]
            else:
                raise NotImplementedError("This dummy backend only supports reduction='none'.")

    print("=" * 60)
    print("Testing LoraBGradientProjector with DummyBackend")
    print("=" * 60)

    dummy_backend = DummyBackend()
    dummy_backend.load()

    projector = LoraBGradientProjector(
        backend=dummy_backend,
        proj_dim=4,
        device="cpu",
        use_cuda_projector=False,
    )

    prompts = ["What is 2+2?"]
    responses = [" 4"]

    token_grads = projector.compute_token_gradients(prompts, responses)
    print(f"Raw token gradients shape: {token_grads[0].shape} (k x grad_dim)")

    projected = projector.compute_and_project(prompts, responses)
    print(f"Projected token gradients shape: {projected[0].shape} (k x proj_dim)")


