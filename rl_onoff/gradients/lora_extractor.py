"""
LoRA gradient extraction and projection utilities.

- LoraBGradientProjector: computes per-token gradients w.r.t. LoRA B matrices
  and projects them to a lower-dimensional space using a Projector
  (BasicProjector or CudaProjector).
"""

from typing import List, Dict, Tuple, Union
import math

import numpy as np
import torch

from rl_onoff.backends.base import BaseBackend
from rl_onoff.gradients.projectors import (
    BasicProjector,
    CudaProjector,
    ChunkedCudaProjector,
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
        proj_dim: int = 8192,
        device: Union[str, torch.device] = "cuda",
        proj_type: str = "rademacher",
        use_cuda_projector: bool = True,
        use_chunk: bool = False,
        block_size: int = 100,
        seed: int = 0,
        cuda_max_batch_size: int = 32,
        chunk_max_size: int = 8192 * 64,
        chunk_feat_bs: int = 2048,
    ) -> None:
        """
        Args:
            backend: HuggingFaceBackend instance with LoRA attached.
            proj_dim: target projection dimension d.
            device: device for projector ("cuda" recommended).
            proj_type: "normal" or "rademacher".
            use_cuda_projector: if True, use CudaProjector-based projector; else BasicProjector.
            use_chunk: if True and use_cuda_projector is True, use ChunkedCudaProjector
                (splits the LoRA-B gradient dimension into chunks projected by multiple
                 CudaProjectors). Ignored if use_cuda_projector is False.
            block_size: block size for BasicProjector (if used).
            seed: random seed for projector.
            cuda_max_batch_size: max batch size for CudaProjector.
            chunk_max_size: maximum number of parameters per chunk for ChunkedCudaProjector.
            chunk_feat_bs: maximum number of tokens (batch size) per ChunkedCudaProjector call.
        """
        self.backend = backend
        self.device = torch.device(device)
        self.proj_dim = proj_dim
        self.proj_type = ProjectionType(proj_type)
        self.seed = seed
        self.use_chunk = bool(use_chunk)

        # Infer grad_dim (total number of LoRA-B parameters) and layout
        self.lora_b_param_names, self.param_shapes, self.grad_dim = (
            self._inspect_lora_b_parameters()
        )

        # Initialize projector
        if use_cuda_projector:
            if self.use_chunk:
                if self.device.type != "cuda":
                    raise ValueError(
                        "ChunkedCudaProjector requires a CUDA device. "
                        "Set device='cuda' or disable use_chunk."
                    )

                max_size = int(chunk_max_size) if chunk_max_size is not None else self.grad_dim
                max_size = max(1, min(max_size, self.grad_dim))

                num_chunks = math.ceil(self.grad_dim / max_size)
                projector_per_chunk = []
                params_per_chunk: List[int] = []
                chunk_slices: List[Tuple[int, int]] = []

                for idx in range(num_chunks):
                    start = idx * max_size
                    end = min((idx + 1) * max_size, self.grad_dim)
                    chunk_dim = end - start
                    chunk_slices.append((start, end))
                    params_per_chunk.append(chunk_dim)

                    projector_per_chunk.append(
                        CudaProjector(
                            grad_dim=chunk_dim,
                            proj_dim=self.proj_dim,
                            seed=self.seed + idx * 1000,
                            proj_type=self.proj_type,
                            device=self.device,
                            max_batch_size=cuda_max_batch_size,
                        )
                    )

                self._chunk_slices = chunk_slices
                feat_bs = int(chunk_feat_bs) if chunk_feat_bs is not None else 2048

                self.projector = ChunkedCudaProjector(
                    projector_per_chunk=projector_per_chunk,
                    max_chunk_size=max_size,
                    params_per_chunk=params_per_chunk,
                    feat_bs=feat_bs,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
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
            if "lora_b" in name.lower():
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
            if isinstance(self.projector, ChunkedCudaProjector):
                if not hasattr(self, "_chunk_slices"):
                    raise RuntimeError(
                        "ChunkedCudaProjector is enabled but _chunk_slices not found. "
                        "This indicates an internal initialization error."
                    )

                k = grads_tensor.shape[0]
                if k > self.projector.feat_bs:
                    raise ValueError(
                        f"Number of tokens ({k}) exceeds ChunkedCudaProjector feat_bs "
                        f"({self.projector.feat_bs}). Increase chunk_feat_bs when "
                        "constructing LoraBGradientProjector."
                    )

                grads_dict: Dict[str, torch.Tensor] = {}
                for idx, (start, end) in enumerate(self._chunk_slices):
                    grads_dict[f"chunk_{idx}"] = grads_tensor[:, start:end]

                projected_tensor = self.projector.project(grads_dict, model_id=model_id)
            else:
                projected_tensor = self.projector.project(grads_tensor, model_id=model_id)

        return projected_tensor.cpu().numpy()

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
    """Simple smoke test for LoraBGradientProjector with a HuggingFace backend."""

    from rl_onoff.backends import create_backend
    from rl_onoff.backends.config import BackendConfig

    print("=" * 60)
    print("Testing LoraBGradientProjector with HuggingFaceBackend")
    print("=" * 60)

    # Configure a small HF model with LoRA adapters.
    # You can change model_name to any causal LM you have locally / can download.
    model_name = "/data/chenyamei/pretrained_models/Qwen3-4B"

    backend_cfg = BackendConfig.from_dict({
        "backend_type": "huggingface",
        "model_name": model_name,
        "backend_specific": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "lora_config": {
                "r": 1,
                "lora_alpha": 32,
                # Apply LoRA to typical attention + MLP projection modules
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                "lora_dropout": 0.1,
            },
        },
    })

    backend = create_backend(backend_cfg)
    backend.load()

    projector = LoraBGradientProjector(
        backend=backend,
        proj_dim=8192,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_cuda_projector=True,  # set True if you have fast_jl installed
        use_chunk=False,
        cuda_max_batch_size = 2
    )

    prompts = ["What is 2+2?"]
    responses = [" 4"*1000]

    token_grads = projector.compute_token_gradients(prompts, responses)
    print(f"Raw token gradients shape: {token_grads[0].shape} (k x grad_dim)")

    projected = projector.compute_and_project(prompts, responses)
    print(f"Projected token gradients shape: {projected[0].shape} (k x proj_dim)")

