import torch
from torch import nn
import torch.nn.functional as F
import math
from copy import deepcopy


def get_torch_device(device):
    if device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


# ------------------------------------------------------------------------------
# ---------------------------------- LAYERS ------------------------------------
# ------------------------------------------------------------------------------

class LinearN(nn.Module):
    """
    Equivalent to N parallel linear layers, each normalized (optionally) independently.
    To be combined with aggregations (max, concat, mean, ...) -- see layers below.

    Shape: (..., in_features) → (..., num_layers, out_features).
    """

    def __init__(
        self,
        in_features,
        out_features,
        num_layers,
        normalization = nn.Identity,
        bias = False,
        init = lambda x : x,
        **kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.linear = nn.Linear(
            in_features,
            out_features * num_layers,
            bias=bias,
        ).apply(init)
        self.norm = normalization(out_features)

    def forward(self, x):
        lead_shape = list(x.size())[:-1]
        out = self.linear(x)
        out = out.view(*lead_shape, self.num_layers, self.out_features)  # (..., num_layers, out_features)
        out = self.norm(out)  # normalizes each of the layers independently
        return out


class Aggregation(nn.Module):
    """
    Concatenates statistics along the aggregation dimension.

    Default shape: (..., num_layers, out_features) → (..., num_stats * out_features).

    Supported stats:
        - "raw" (concatenates all features, no operation)
        - "mean"
        - "prod"
        - "max"
        - "min"
        - "std"

    Aggregation(["max"]) would give
        Maxout (Goodfellow et al., 2013).

    Aggregation(["mean", "std", "max", "min"]) would give
        Principal Neighbourhood Aggregation (Corso et al., 2020).
    """

    def __init__(self, stats, dim=-2, **kwargs):
        super().__init__()
        self.stats = stats
        self.dim = dim

    def forward(self, x):
        outputs = []

        for stat in self.stats:
            if stat == "raw":
                outputs.append(x)
            elif stat == "mean":
                outputs.append(x.mean(dim=self.dim, keepdim=True))
            elif stat == "prod":
                outputs.append(x.prod(dim=self.dim, keepdim=True))
            elif stat == "max":
                outputs.append(x.amax(dim=self.dim, keepdim=True))
            elif stat == "min":
                outputs.append(x.amin(dim=self.dim, keepdim=True))
            elif stat == "std":
                outputs.append(x.std(dim=self.dim, keepdim=True, unbiased=False))
            else:
                raise ValueError(f"Unknown statistic: {stat}")

        return torch.cat(outputs, dim=self.dim).flatten(
            start_dim=self.dim, end_dim=self.dim + 1
        )


class FuzzyTileActivation(nn.Module):
    """
    Fuzzy Tiling Activation (Pan et al., 2021, arXiv:1911.08068).

    Sparse tile-coded representation. Each tile emits 1 when the input lies
    inside it and ramps to 0 over a fuzzy band of width `eta` (= eta_ratio *
    tile width) outside it.

    centers_mode:
      "fixed"  (default): tile edges `c` are fixed buffers; gradient flows
                          through the input, never into the tiling. Intended
                          to sit after a bounded/normalized pre-activation
                          (Tanh, clamped LayerNorm) so inputs stay in
                          [v_min, v_max]. Outside that range every tile
                          saturates to zero → a clean sparse code.
      "learned": `c` is an nn.Parameter trained end-to-end.

    boundary_leak (defaults to True iff centers_mode == "learned"):
      Adds a linear leak on the two outermost tiles outside the domain --
      constant gradient of -1 left of the first edge, +1 right of the last
      edge -- keeping a live gradient into `c` for inputs arbitrarily far
      outside [v_min, v_max]. This is what lets learned edges migrate back
      toward unbounded data. It is OFF in fixed mode by default, so fixed
      mode's saturate-to-zero property actually holds; pass boundary_leak=True
      to force it on regardless.
    """

    CENTERS_MODES = ("fixed", "learned")

    def __init__(
        self,
        in_features,
        num_centers,
        v_min = None,
        v_max = None,
        eta_ratio = 0.2,
        centers_mode = "fixed",
        boundary_leak = None,
        **kwargs,
    ):
        super().__init__()
        assert centers_mode in self.CENTERS_MODES, \
            f"centers_mode must be one of {self.CENTERS_MODES}"
        assert not (v_min is None and v_max is None), \
            "must define at least v_min or v_max"
        if v_max is None:
            v_max = -v_min
        elif v_min is None:
            v_min = -v_max

        self.in_features = in_features
        self.num_centers = num_centers
        self.centers_mode = centers_mode
        self.boundary_leak = (centers_mode == "learned") if boundary_leak is None \
            else bool(boundary_leak)

        v_min = torch.asarray(v_min) * torch.ones(in_features)
        v_max = torch.asarray(v_max) * torch.ones(in_features)
        delta = (v_max - v_min) / num_centers
        steps = torch.arange(num_centers)
        c = v_min[:, None] + steps[None, :] * delta[:, None]  # left edge of each tile

        if centers_mode == "learned":
            self.c = nn.Parameter(c.unsqueeze(0))
        else:
            self.register_buffer("c", c.unsqueeze(0))
        self.register_buffer("delta", delta.view(1, in_features, 1))
        self.register_buffer("eta", (eta_ratio * delta).view(1, in_features, 1))

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, self.in_features, 1)

        # `outside` is the distance the input lies past the tile edges (0 while
        # inside [c, c + delta)). The activation is therefore 1 inside the tile,
        # ramps down to 0 over the next `eta`, and stays at 0 beyond that.
        outside = (torch.clamp(self.c - x, min=0)
                   + torch.clamp(x - (self.c + self.delta), min=0))
        activation = 1.0 - torch.clamp(outside / self.eta, max=1.0)

        if self.boundary_leak:
            # Distance past the domain edges, routed to the corner tiles.
            # Padding into a full-width leak tensor (instead of in-place slice
            # assignment) stays differentiable, avoids a clone, and makes the
            # two leaks *sum* when num_centers == 1 (both corners collapse onto
            # the same column).
            left = torch.clamp(self.c[:, :, :1] - x, min=0)                  # (B, F, 1)
            right = torch.clamp(x - (self.c[:, :, -1:] + self.delta), min=0)  # (B, F, 1)
            t = self.num_centers
            leak = F.pad(left, (0, t - 1)) + F.pad(right, (t - 1, 0))        # (B, F, T)
            activation = activation + leak

        return activation.reshape(*shp[:-1], -1)


class GaussianActivation(nn.Module):
    """
    Separable per-dimension Gaussian activation: one independent 1D Gaussian
    basis per input dimension (not a joint multivariate RBF over the full
    input vector). Each unit emits a Gaussian bump centered on `mu`; the
    bandwidth `sigma` is initialized to the center spacing so adjacent units
    overlap, and floored at `sigma_min` (in "learned" mode it is trained via
    softplus over `raw_sigma`). This is a dense encoding -- every unit is
    always nonzero, so it carries no hard sparsity guarantee.

    centers_mode:
      "fixed"  (default): centers/bandwidths `mu`/`sigma` are fixed buffers;
                          gradient flows through the input, never into the
                          Gaussian. Intended to sit after a bounded/normalized
                          pre-activation (Tanh, LayerNorm) so inputs stay in
                          [v_min, v_max]. Outside that range every Gaussian
                          decays toward zero (except the leaked corner units,
                          below).
      "learned": `mu`/`raw_sigma` are nn.Parameters trained end-to-end.

    boundary_leak:
      The two outermost units carry a linear leak outside the domain (added
      on top of their Gaussian activation) -- constant gradient of -1 left
      of the first center, +1 right of the last center -- keeping a live
      gradient into `mu` for inputs arbitrarily far outside [v_min, v_max].
      This is always on, in both modes; it is what lets learned centers
      migrate back toward unbounded data. Note the leaked corner units grow
      past 1 and are no longer peaked at their center, so the output is not
      bounded to [0, 1].
    """

    CENTERS_MODES = ("fixed", "learned")

    def __init__(
        self,
        in_features,
        num_centers,
        v_min = None,
        v_max = None,
        centers_mode = "fixed",
        sigma_min = 1e-3,
        **kwargs,
    ):
        super().__init__()
        assert centers_mode in self.CENTERS_MODES, \
            f"centers_mode must be one of {self.CENTERS_MODES}"
        assert not (v_min is None and v_max is None), \
            "must define at least v_min or v_max"
        assert num_centers >= 2, \
            "num_centers must be >= 2 (center spacing needs at least two centers)"
        if v_max is None:   v_max = -v_min
        elif v_min is None: v_min = -v_max

        self.in_features = in_features
        self.num_centers = num_centers
        self.centers_mode = centers_mode
        self.sigma_min = sigma_min

        v_min = torch.asarray(v_min) * torch.ones(in_features)
        v_max = torch.asarray(v_max) * torch.ones(in_features)
        steps = torch.linspace(0, 1, num_centers)
        mu_init = v_min[:, None] + steps[None, :] * (v_max - v_min)[:, None]
        dist = (v_max - v_min) / (num_centers - 1)
        dist = dist.view(1, in_features, 1).expand(1, in_features, num_centers).clone()

        if centers_mode == "learned":
            self.mu = nn.Parameter(mu_init.unsqueeze(0))
            # Stable inverse softplus: log(exp(dist) - 1) rewritten so exp()
            # never overflows for wide domains -- dist + log(1 - exp(-dist)).
            self.raw_sigma = nn.Parameter(dist + torch.log(-torch.expm1(-dist)))
        else:
            self.register_buffer("mu", mu_init.unsqueeze(0))
            self.register_buffer("sigma", dist.clamp(min=sigma_min))

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, self.in_features, 1)

        # Bandwidth is floored at sigma_min in both modes: fixed clamps at
        # build time, learned clamps the softplus output every forward so a
        # trained Gaussian can't shrink to a degenerate spike.
        if self.centers_mode == "learned":
            sigma = F.softplus(self.raw_sigma).clamp(min=self.sigma_min)
        else:
            sigma = self.sigma

        diff = x - self.mu
        activation = torch.exp(-0.5 * (diff / (sigma + 1e-8)) ** 2)

        # Linear out-of-bounds leak on the two outermost Gaussians: outside the
        # domain the Gaussians vanish, so add a constant +/-1-gradient ramp
        # to keep a live signal into the input (and into `mu` in learned
        # mode). Padding into a full-width tensor (instead of in-place slice
        # assignment) stays differentiable and avoids a clone; num_centers >= 2
        # guarantees the two corner columns never collide. Note the corner
        # Gaussian's value then exceeds 1 and is no longer peaked at its center --
        # output range is not bounded to [0, 1].
        left_leak = torch.clamp(self.mu[:, :, :1] - x, min=0)    # (B, F, 1)
        right_leak = torch.clamp(x - self.mu[:, :, -1:], min=0)  # (B, F, 1)
        t = self.num_centers
        leak = F.pad(left_leak, (0, t - 1)) + F.pad(right_leak, (t - 1, 0))  # (B, F, T)
        activation = activation + leak

        return activation.reshape(*shp[:-1], -1)


class OutNorm(nn.Module):
    """Normalize along the last dimension."""
    def __init__(self, p=2, eps=1e-12):
        super(OutNorm, self).__init__()
        self.eps = eps
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=-1, eps=self.eps)


class SymLog(nn.Module):
    """https://arxiv.org/abs/2301.04104"""
    def __init__(self, clamp=None):
        super().__init__()
        self.clamp = clamp

    def forward(self, x):
        if self.clamp is not None:
            x = torch.clamp(x, -self.clamp, self.clamp)
        return torch.sign(x) * torch.log1p(torch.abs(x))


class SymExp(nn.Module):
    """https://arxiv.org/abs/2301.04104"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class Swish(nn.Module):
    """https://arxiv.org/abs/1606.08415"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class Elephant(nn.Module):
    """https://arxiv.org/abs/2310.01365"""
    def __init__(self, a: float = 1.0, h: float = 1.0, d: float = 2.0):
        super(Elephant, self).__init__()
        self.a = a
        self.h = h
        self.d = d

    def forward(self, x):
        abs_x = torch.abs(x / self.a)
        return self.h / (1.0 + abs_x ** self.d)


class FlattenBatch(nn.Module):
    """Flattens any dimension not related to the input."""
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape  # e.g., (C, H, W)

    def forward(self, x):
        return x.reshape(-1, *self.input_shape)


class LazyLayerNorm(nn.Module):
    """Lazy version of LayerNorm (no need to manually specify input size)."""
    def forward(self, input):
        return F.layer_norm(input, input.size())


class LazyRMSNorm(nn.Module):
    """Lazy version of RMSNorm (no need to manually specify input size)."""
    def forward(self, input):
        return F.rms_norm(input, input.size())


class MyBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that works when the input has only one dimension."""
    def forward(self, x):
        return super().forward(x.view(-1, x.shape[-1])).view_as(x)



# ------------------------------------------------------------------------------
# ----------------------------------- INIT -------------------------------------
# ------------------------------------------------------------------------------

def zero_init(module):
    if module.weight is not None:
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
    if module.bias is not None:
        module.bias.data.fill_(0.0)


def xavier_init(module, activation):
    if module.weight is not None:
        nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain(activation))
    if module.bias is not None:
        module.bias.data.fill_(0.0)


def sparse_init(module, sparsity, type='uniform'):
    if module.bias is not None:
        module.bias.data.fill_(0.0)

    if module.weight is not None:
        tensor = module.weight
        fan_out, fan_in = tensor.shape
        num_zeros = int(math.ceil(sparsity * fan_in))

        with torch.no_grad():
            if type == "uniform":
                tensor.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
            elif type == "normal":
                tensor.normal_(0, math.sqrt(1.0 / fan_in))
            else:
                raise ValueError("Unknown initialization type")
            for col_idx in range(fan_out):
                row_indices = torch.randperm(fan_in)
                zero_indices = row_indices[:num_zeros]
                tensor[col_idx, zero_indices] = 0


# ------------------------------------------------------------------------------
# ---------------------------------- RESETS ------------------------------------
# ------------------------------------------------------------------------------

@torch.no_grad()
def add_noise_to_parameters(module, std):
    for param in module.parameters():
        noise = torch.randn(
            param.shape,
            device=param.device,
            dtype=param.dtype,
        ) * std
        param.add_(noise)


def soft_reset(module, alpha: float, seed: int = None):
    """
    Ash & Adams (2020) shrink-and-perturb, in-place on `module`:
        θ ← α · θ_old + (1 − α) · θ_fresh
    α = 1 keeps `module` unchanged; α = 0 fully replaces it with the fresh
      init. Typical soft-reset values: α ∈ [0.5, 0.9].
    Both parameters and buffers (running stats, etc.) are blended.
    """

    fresh_module = deepcopy(module)
    fresh_module.reset(seed=seed)
    for p, p_fresh in zip(module.parameters(), fresh_module.parameters()):
        p.data.copy_(alpha * p.data + (1.0 - alpha) * p_fresh.data)
    for b, b_fresh in zip(module.buffers(), fresh_module.buffers()):
        b.data.copy_(alpha * b.data + (1.0 - alpha) * b_fresh.data)


def soft_reset_optimizer(optimizer, alpha: float):
    """
    Soft reset optimizer state: blend old optimizer state with zero state.
    α = 1 keeps the state unchanged; α = 0 fully resets to zero state.
    Typical soft-reset values: α ∈ [0.5, 0.9].
    """

    old_state = deepcopy(optimizer.state_dict())
    for state_key, state_dict in old_state.get('state', {}).items():
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = alpha * v + (1.0 - alpha) * torch.zeros_like(v)
            else:
                state_dict[k] = v
    optimizer.load_state_dict(old_state)
