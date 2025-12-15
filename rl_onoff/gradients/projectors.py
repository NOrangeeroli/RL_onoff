"""
Random projection utilities for gradient vectors.

Implements:
- NoOpProjector
- BasicSingleBlockProjector
- BasicProjector
- CudaProjector
- ChunkedCudaProjector
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Dict, List, Any
import math

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Small helper: vectorize a dict of parameter tensors into a 2D tensor
# grads: {name -> Tensor of shape (batch, ...)} → Tensor (batch, grad_dim)
# ---------------------------------------------------------------------------


def vectorize(grads: Dict[str, Tensor], device: Union[str, torch.device]) -> Tensor:
    """Flatten and concatenate a dict of tensors into a single 2D tensor.

    Assumes all tensors have batch size on dim 0.
    """
    if not grads:
        raise ValueError("Empty gradient dict passed to vectorize().")

    flats: List[Tensor] = []
    batch_size = None
    for p in grads.values():
        if batch_size is None:
            batch_size = p.shape[0]
        else:
            assert p.shape[0] == batch_size, "All grads must share batch dimension in dim 0."

        if p.dim() == 1:
            flat = p.view(batch_size, 1)
        else:
            flat = p.view(batch_size, -1)
        flats.append(flat)

    out = torch.cat(flats, dim=1)  # (batch, grad_dim)
    return out.to(device)


# ---------------------------------------------------------------------------
# Core types and abstract base
# ---------------------------------------------------------------------------

ch = torch


class ProjectionType(str, Enum):
    normal: str = "normal"
    rademacher: str = "rademacher"


class AbstractProjector(ABC):
    """Base class for all projectors."""

    @abstractmethod
    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: Union[str, ProjectionType],
        device: Union[str, torch.device],
    ) -> None:
        """
        Args:
            grad_dim: number of parameters (dimension of gradient vectors)
            proj_dim: dimension after projection
            seed: random seed
            proj_type: "normal" or "rademacher"
            device: CUDA device or CPU
        """
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.proj_type = proj_type
        self.device = device

    @abstractmethod
    def project(self, grads: Union[Dict[str, Tensor], Tensor], model_id: int) -> Tensor:
        """Project a batch of gradients.

        Args:
            grads: Tensor of shape (batch, grad_dim) OR
                   dict{name->Tensor}, which will be vectorized
            model_id: unique id to allow different projection matrices

        Returns:
            Tensor of shape (batch, proj_dim)
        """

    def free_memory(self) -> None:
        """Frees up any memory used by the projector."""
        pass


# ---------------------------------------------------------------------------
# No-op projector
# ---------------------------------------------------------------------------


class NoOpProjector(AbstractProjector):
    """Projector that returns gradients unchanged."""

    def __init__(
        self,
        grad_dim: int = 0,
        proj_dim: int = 0,
        seed: int = 0,
        proj_type: Union[str, ProjectionType] = "na",
        device: Union[str, torch.device] = "cuda",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

    def project(self, grads: Union[Dict[str, Tensor], Tensor], model_id: int) -> Tensor:
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)
        return grads

    def free_memory(self) -> None:
        pass


# ---------------------------------------------------------------------------
# BasicSingleBlockProjector
# ---------------------------------------------------------------------------


class BasicSingleBlockProjector(AbstractProjector):
    """Bare-bones, single-block projection using matmul."""

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device: Union[str, torch.device],
        dtype: torch.dtype = ch.float32,
        model_id: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.model_id = model_id
        self.proj_type = proj_type
        self.generator = ch.Generator(device=self.device)
        self.generator = self.generator.manual_seed(
            self.seed + int(1e4) * self.model_id
        )
        self.dtype = dtype

        self.proj_matrix = ch.empty(
            self.grad_dim, self.proj_dim, dtype=self.dtype, device=self.device
        )

        self.proj_matrix_available = True
        self.generate_sketch_matrix()

    def free_memory(self) -> None:
        del self.proj_matrix
        self.proj_matrix_available = False

    def generate_sketch_matrix(self) -> None:
        if not self.proj_matrix_available:
            self.proj_matrix = ch.empty(
                self.grad_dim, self.proj_dim, dtype=self.dtype, device=self.device
            )
            self.proj_matrix_available = True

        if self.proj_type == ProjectionType.normal or self.proj_type == "normal":
            self.proj_matrix.normal_(generator=self.generator)
        elif self.proj_type == ProjectionType.rademacher or self.proj_type == "rademacher":
            self.proj_matrix.bernoulli_(p=0.5, generator=self.generator)
            self.proj_matrix *= 2.0
            self.proj_matrix -= 1.0
        else:
            raise KeyError(f"Projection type {self.proj_type} not recognized.")
        # Scale by 1/sqrt(k) to approximate JL isotropy (k = proj_dim)
        self.proj_matrix /= math.sqrt(self.proj_dim)

    def project(self, grads: Union[Dict[str, Tensor], Tensor], model_id: int) -> Tensor:
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)

        grads = grads.to(dtype=self.dtype)
        if model_id != self.model_id:
            self.model_id = model_id
            self.generator = self.generator.manual_seed(
                self.seed + int(1e4) * self.model_id
            )
            self.generate_sketch_matrix()

        return grads @ self.proj_matrix


# ---------------------------------------------------------------------------
# BasicProjector (block-wise)
# ---------------------------------------------------------------------------


class BasicProjector(AbstractProjector):
    """Block-wise projection: generates projection matrix in blocks."""

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device: Union[str, torch.device],
        block_size: int = 100,
        dtype: torch.dtype = ch.float32,
        model_id: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)

        self.block_size = min(self.proj_dim, block_size)
        self.num_blocks = math.ceil(self.proj_dim / self.block_size)
        self.dtype = dtype
        self.proj_type = proj_type
        self.model_id = model_id

        self.proj_matrix = ch.empty(
            self.grad_dim, self.block_size, dtype=self.dtype, device=self.device
        )

        self.proj_matrix_available = True

        self.generator = ch.Generator(device=self.device)

        self.get_generator_states()
        self.generate_sketch_matrix(self.generator_states[0])

    def free_memory(self) -> None:
        del self.proj_matrix
        self.proj_matrix_available = False

    def get_generator_states(self) -> None:
        self.generator_states: List[Tensor] = []
        self.seeds: List[int] = []
        self.jl_size = self.grad_dim * self.block_size

        for i in range(self.num_blocks):
            s = self.seed + int(1e3) * i + int(1e5) * self.model_id
            self.seeds.append(s)
            self.generator = self.generator.manual_seed(s)
            self.generator_states.append(self.generator.get_state())

    def generate_sketch_matrix(self, generator_state: Tensor) -> None:
        if not self.proj_matrix_available:
            self.proj_matrix = ch.empty(
                self.grad_dim, self.block_size, dtype=self.dtype, device=self.device
            )
            self.proj_matrix_available = True

        self.generator.set_state(generator_state)
        if self.proj_type == ProjectionType.normal or self.proj_type == "normal":
            self.proj_matrix.normal_(generator=self.generator)
        elif self.proj_type == ProjectionType.rademacher or self.proj_type == "rademacher":
            self.proj_matrix.bernoulli_(p=0.5, generator=self.generator)
            self.proj_matrix *= 2.0
            self.proj_matrix -= 1.0
        else:
            raise KeyError(f"Projection type {self.proj_type} not recognized.")
        # Scale by 1/sqrt(k) to approximate JL isotropy (k = total proj_dim)
        self.proj_matrix /= math.sqrt(self.proj_dim)


# ---------------------------------------------------------------------------
# Johnson–Lindenstrauss style sanity checks for projectors (CPU, BasicProjector)
# ---------------------------------------------------------------------------


def jl_min_k(n: int, eps: float) -> int:
    """Minimum JL dimension k for n points and distortion eps (theoretical bound)."""
    return int(math.ceil(4 * math.log(n) / (eps ** 2 / 2.0 - eps ** 3 / 3.0)))


def sanity_checks_projector(
    n: int,
    d: int,
    k: int,
    eps: float = 0.2,
    pairs: int = 20_000,
    seed: int = 0,
    proj_type: ProjectionType = ProjectionType.rademacher,
    device: Union[str, torch.device] = "cpu",
) -> None:
    """Run JL-style sanity checks for BasicProjector on random Gaussian data.

    This mirrors the numpy reference code:
      - checks output shape
      - checks approximate isotropy of P^T P
      - checks norm preservation per point
      - checks pairwise distance distortions on random pairs
    """
    device = torch.device(device)
    rng = torch.Generator(device=device).manual_seed(seed)

    # Random Gaussian data X ~ N(0, I_d)
    X = torch.randn(n, d, generator=rng, device=device)

    # Use BasicProjector with a single block (block_size == k) to get an explicit P
    proj = BasicProjector(
        grad_dim=d,
        proj_dim=k,
        seed=seed,
        proj_type=proj_type,
        device=device,
        block_size=k,
    )
    Y = proj.project(X, model_id=0)  # (n, k)

    print("=== Sanity checks for BasicProjector ===")
    print(f"n={n}, d={d}, k={k}, eps={eps}, proj_type={proj_type}")

    # A) shape check
    assert Y.shape == (n, k), f"Y shape mismatch: got {Y.shape}, expected {(n, k)}"
    print("Shape check passed:", Y.shape)

    # B) Isotropy-ish check: P^T P ≈ I_k
    # BasicProjector.proj_matrix is of shape (d, k) since block_size == k
    P = proj.proj_matrix  # (d, k)
    G = P.t() @ P  # (k, k)
    diag = torch.diag(G)
    diag_err = (diag - 1.0).abs().mean().item()
    offdiag = G - torch.diag(diag)
    offdiag_mean = offdiag.abs().sum().item() / (k * k - k)
    print("mean |diag(P^T P) - 1|:", diag_err)
    print("mean |offdiag(P^T P)|:", offdiag_mean)

    # C) Norm ratios on random rows
    xnorm2 = (X * X).sum(dim=1)
    ynorm2 = (Y * Y).sum(dim=1)
    mask = xnorm2 > 0
    ratios = (ynorm2[mask] / xnorm2[mask]).detach().cpu().numpy()
    qs = np.quantile(ratios, [0.01, 0.1, 0.5, 0.9, 0.99])
    print("norm ratio quantiles (Y^2 / X^2):", qs)

    # D) Pairwise distortion on sampled pairs
    rng_pairs = torch.Generator(device=device).manual_seed(seed + 1)
    idx_i = torch.randint(0, n, (pairs,), generator=rng_pairs, device=device)
    idx_j = torch.randint(0, n, (pairs,), generator=rng_pairs, device=device)

    dx = X[idx_i] - X[idx_j]
    dy = Y[idx_i] - Y[idx_j]
    dist_x = dx.norm(dim=1)
    dist_y = dy.norm(dim=1)
    ok = dist_x > 1e-12
    distort = (dist_y[ok] / dist_x[ok]).detach().cpu().numpy()

    qs_dist = np.quantile(distort, [0.01, 0.1, 0.5, 0.9, 0.99])
    frac_in = np.mean((distort >= 1 - eps) & (distort <= 1 + eps))
    print("pairwise distortion quantiles (||Px-Py|| / ||x-y||):", qs_dist)
    print(f"fraction within [1±eps] (eps={eps}):", frac_in)

    def project(self, grads: Union[Dict[str, Tensor], Tensor], model_id: int) -> Tensor:
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)

        grads = grads.to(dtype=self.dtype)
        sketch = ch.zeros(
            size=(grads.size(0), self.proj_dim), dtype=self.dtype, device=self.device
        )

        if model_id != self.model_id:
            self.model_id = model_id
            self.get_generator_states()
            if self.num_blocks == 1:
                self.generate_sketch_matrix(self.generator_states[0])

        if self.num_blocks == 1:
            ch.matmul(grads.data, self.proj_matrix, out=sketch)
        else:
            for ind in range(self.num_blocks):
                self.generate_sketch_matrix(self.generator_states[ind])

                st = ind * self.block_size
                ed = min((ind + 1) * self.block_size, self.proj_dim)
                sketch[:, st:ed] = (
                    grads.type(self.dtype) @ self.proj_matrix[:, : (ed - st)]
                )
        return sketch.type(grads.dtype)


# ---------------------------------------------------------------------------
# CudaProjector
# ---------------------------------------------------------------------------


class CudaProjector(AbstractProjector):
    """CUDA-based JL projector using fast_jl."""

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device: Union[str, torch.device],
        max_batch_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)
        self.max_batch_size = max_batch_size

        if isinstance(device, str):
            device = ch.device(device)

        if device.type != "cuda":
            err = (
                "CudaProjector only works on a CUDA device; either switch to a CUDA "
                "device, or use the BasicProjector"
            )
            raise ValueError(err)

        self.num_sms = ch.cuda.get_device_properties(device.index).multi_processor_count

        try:
            import fast_jl  # noqa: F401

            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(
                ch.zeros(8, 1_000, device="cuda"), 512, 0, self.num_sms
            )
        except ImportError:
            err = (
                "You should make sure to install the CUDA projector for TRAKer "
                "(called fast_jl). See installation docs for details."
            )
            raise ModuleNotFoundError(err)

    def project(
        self,
        grads: Union[Dict[str, Tensor], Tensor],
        model_id: int,
    ) -> Tensor:
        """Project a batch of gradients using the CUDA fast_jl kernels."""
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)

        batch_size = grads.shape[0]

        # Choose an effective batch size compatible with fast_jl kernels
        effective_batch_size = 32
        if batch_size <= 8:
            effective_batch_size = 8
        elif batch_size <= 16:
            effective_batch_size = 16

        effective_batch_size = min(self.max_batch_size, effective_batch_size)

        function_name = f"project_{self.proj_type.value}_{effective_batch_size}"
        import fast_jl

        fn = getattr(fast_jl, function_name)

        try:
            result = fn(
                grads, self.proj_dim, self.seed + int(1e4) * model_id, self.num_sms
            )
        except RuntimeError as e:
            if "CUDA error: too many resources requested for launch" in str(e):
                raise RuntimeError(
                    "The batch size of the CudaProjector is too large for your GPU. "
                    "Reduce it by using a smaller max_batch_size.\n"
                    f"Original error: {e}"
                )
            else:
                raise e

        return result


# ---------------------------------------------------------------------------
# ChunkedCudaProjector (optional helper for very large parameter vectors)
# ---------------------------------------------------------------------------


class ChunkedCudaProjector:
    """Chunked wrapper around multiple CudaProjectors for very large grad_dim.

    This is useful when a single CudaProjector would require too much memory
    to handle all parameters at once. Instead, we split the gradient vector
    into chunks and project each chunk with its own CudaProjector.
    """

    def __init__(
        self,
        projector_per_chunk: List[CudaProjector],
        max_chunk_size: int,
        params_per_chunk: List[int],
        feat_bs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.projector_per_chunk = projector_per_chunk
        self.proj_dim = self.projector_per_chunk[0].proj_dim
        self.proj_type = self.projector_per_chunk[0].proj_type
        self.params_per_chunk = params_per_chunk

        self.max_chunk_size = max_chunk_size
        self.feat_bs = feat_bs
        self.device = device
        self.dtype = dtype
        self.input_allocated = False

    def allocate_input(self) -> None:
        if self.input_allocated:
            return

        self.ch_input = ch.zeros(
            size=(self.feat_bs, self.max_chunk_size),
            device=self.device,
            dtype=self.dtype,
        )

        self.input_allocated = True

    def free_memory(self) -> None:
        if not self.input_allocated:
            return

        del self.ch_input
        self.input_allocated = False

    def project(self, grads: Dict[str, Tensor], model_id: int) -> Tensor:
        """Project a dict of parameter tensors using multiple CUDA projectors."""
        self.allocate_input()
        ch_output = ch.zeros(
            size=(self.feat_bs, self.proj_dim), device=self.device, dtype=self.dtype
        )
        pointer = 0
        projector_index = 0

        # Iterate over params, chunking by max_chunk_size
        for _, p in grads.items():
            if len(p.shape) < 2:
                p_flat = p.data.unsqueeze(-1)
            else:
                p_flat = p.data.flatten(start_dim=1)

            param_size = p_flat.size(1)
            if pointer + param_size > self.max_chunk_size:
                assert pointer == self.params_per_chunk[projector_index]
                ch_output.add_(
                    self.projector_per_chunk[projector_index].project(
                        self.ch_input[:, :pointer].contiguous(),
                        model_id=model_id,
                    )
                )
                pointer = 0
                projector_index += 1

            actual_bs = min(self.ch_input.size(0), p_flat.size(0))
            self.ch_input[:actual_bs, pointer : pointer + param_size].copy_(p_flat)
            pointer += param_size

        # Project remaining items
        assert pointer == self.params_per_chunk[projector_index]
        ch_output[:actual_bs].add_(
            self.projector_per_chunk[projector_index].project(
                self.ch_input[:actual_bs, :pointer].contiguous(),
                model_id=model_id,
            )
        )

        return ch_output[:actual_bs]


if __name__ == "__main__":
    """Simple smoke tests and sanity checks for projector implementations."""

    print("=" * 60)
    print("Sanity checks for BasicProjector (JL-style)")
    print("=" * 60)
    n = 2000
    d = 2000
    eps = 0.2
    k = jl_min_k(n, eps)
    print(f"Using n={n}, d={d}, k={k} (jl_min_k)")
    sanity_checks_projector(
        n=n,
        d=d,
        k=k,
        eps=eps,
        pairs=20_000,
        seed=123,
        proj_type=ProjectionType.rademacher,
        device="cpu",
    )

    # Simple shape and vectorize checks
    print("\n" + "=" * 60)
    print("Basic shape and vectorize tests")
    print("=" * 60)

    grad_dim = 1024
    proj_dim = 512
    batch_size = 3

    grads = torch.randn(batch_size, grad_dim)
    basic_proj = BasicProjector(
        grad_dim=grad_dim,
        proj_dim=proj_dim,
        seed=42,
        proj_type=ProjectionType.rademacher,
        device="cpu",
        block_size=8,
    )
    projected = basic_proj.project(grads, model_id=0)
    print(f"BasicProjector output shape: {projected.shape} (expected {batch_size} x {proj_dim})")

    # Test vectorize + NoOpProjector with dict input
    print("\n" + "=" * 60)
    print("Testing NoOpProjector with dict of gradients")
    print("=" * 60)
    grads_dict = {
        "w1": torch.randn(batch_size, 5),
        "w2": torch.randn(batch_size, 7),
    }
    noop = NoOpProjector()
    vec = noop.project(grads_dict, model_id=0)
    print(f"NoOpProjector / vectorize output shape: {vec.shape} (expected {batch_size} x 12)")

    # Optional CUDA / fast_jl smoke test (only runs if CUDA and fast_jl are available)
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Testing CudaProjector L2 distance preservation (high dimension)")
        print("=" * 60)
        try:
            import torch.nn.functional as F

            # Original high-dimensional space: 8192 * 64
            orig_dim = 8192 * 64  # 524,288
            proj_dim_cuda = 8192  # must be multiple of 512 for fast_jl
            batch_size_cuda = 16  # number of vectors (including query)

            x = torch.randn(batch_size_cuda, orig_dim, device="cuda")

            # Choose a query vector (e.g., first one)
            q_idx = 0
            q = x[q_idx : q_idx + 1]
            # L2 distances in original space
            dists_orig = torch.cdist(q, x).squeeze(0)

            cuda_proj = CudaProjector(
                grad_dim=orig_dim,
                proj_dim=proj_dim_cuda,
                seed=123,
                proj_type=ProjectionType.rademacher,
                device="cuda",
                max_batch_size=batch_size_cuda,
            )
            x_proj = cuda_proj.project(x, model_id=0)
            # L2 distances in projected space
            dists_proj = F.pairwise_distance(
                x_proj[q_idx : q_idx + 1].expand_as(x_proj), x_proj, p=2
            )

            k = 5

            # Exclude self (index 0) when computing nearest neighbors by smallest L2 distance
            def topk_with_dists(d: torch.Tensor):
                vals, idx = torch.topk(d, k + 1, largest=False)
                # drop self (distance 0 at index 0)
                return idx[1:], vals[1:]

            idx_orig, vals_orig = topk_with_dists(dists_orig)
            idx_proj, vals_proj = topk_with_dists(dists_proj)

            top_orig = idx_orig.tolist()
            top_proj = idx_proj.tolist()

            overlap = len(set(top_orig) & set(top_proj))
            print(f"Top-{k} nearest neighbors (original space) indices: {top_orig}")
            print(f"Top-{k} L2 distances (original): {vals_orig.tolist()}")
            print(f"Top-{k} nearest neighbors (projected space) indices: {top_proj}")
            print(f"Top-{k} L2 distances (projected): {vals_proj.tolist()}")
            print(f"Overlap in top-{k} neighbors: {overlap}/{k}")

        except Exception as e:  # noqa: BLE001
            print(f"Skipping CudaProjector similarity test due to error: {e}")
