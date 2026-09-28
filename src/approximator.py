import numpy as np
from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear

import src.utils.torch_nn
from src.utils.torch_nn import (
    FlattenBatch,
    LinearN,
    Aggregation,
    get_torch_device,
    xavier_init,
    sparse_init,
    zero_init,
)


def get_layer(name):
    if name is None or name == "None":
        return nn.Identity
    if hasattr(src.utils.torch_nn, name):
        return getattr(src.utils.torch_nn, name)
    elif hasattr(nn, name):
        return getattr(nn, name)
    else:
        raise ValueError(f"unknown layer: {name}")


class FunctionApproximator(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def reset(self, seed=None):
        pass

    @abstractmethod
    def copy_from(self):
        pass


class Table(FunctionApproximator):
    """
    Multi-dimensional np.array for discrete inputs.
    When called, it expects inputs (such as states and actions) passed as either
    integers, list of integers, or np.array.
    For example, the following code initializes a Q-function for an MDP with
    9 states and 4 actions, and then asks for the value of three state-action pairs.

        >>> q_fun = Table(9, 4)
        >>> q_fun.shape
        (9, 4)
        >>> state = np.array([0, 1, 2])
        >>> value = q_fun(state)
        >>> value.shape
        (3, 4)
        >>> action = np.array([2, 3, 0])
        >>> value = q_fun(state, action)
        >>> value.shape
        (3,)

    This class also works with multi-dimensional states and actions. In the
    following example, both the state and the action are 2-dimensional, with 9x2
    and 4x2 states and actions, respectively.

        >>> q_fun = Table(9, 2, 4, 2)
        >>> q_fun.shape
        (9, 2, 4, 2)
        >>> state_env = np.array([3, 1, 2])
        >>> state_ext = np.array([0, 1, 0])
        >>> action_env = np.array([2, 3, 0])
        >>> action_ext = np.array([1, 0, 0])
        >>> value = q_fun(state_env, state_ext, action_env, action_ext)
        >>> value.shape
        (3,)
    """

    def __init__(
        self,
        *shape,
        init_value_min: float = 0.0,
        init_value_max: float = 0.0,
        map_shape: tuple = None,
        seed: int = None,
        **kwargs,
    ):
        """
        Args:
            shape (int...): a sequence of integers defining the shape of the table,
            init_value_min (float): the initial table is filled according to a
                uniform distribution in [init_value_min, init_value_max],
            init_value_max (float): see above,
            map_shape (tuple): optional 2D (or n-D) obs-side layout used only
                as a display hint (e.g. gridworld `(H, W)` for reshaping a
                flat `(n_obs,)` count into a heatmap). Not used by the table
                itself. None means no layout hint,
            seed (int): seed to initialize the table for reproducibility,
        """

        self.shape = shape
        self._map_shape = tuple(map_shape) if map_shape is not None else None
        assert init_value_min <= init_value_max, "invalid initialization"
        self._init_value_min = init_value_min
        self._init_value_max = init_value_max
        self.reset(seed=seed)

    def __call__(self, *args):
        if not args:
            return self._table.copy()
        not_none_args = [arg for arg in args if arg is not None]
        if len(not_none_args) == 0:
            return self._table.copy()
        return self._table[tuple(np.stack(not_none_args).astype(int))].copy()

    def update(self, *args, target=None, **kwargs):
        self._table[tuple(np.stack(args).astype(int))] = target

    def reset(self, seed=None):
        self._table = np.random.default_rng(seed=seed).uniform(
            self._init_value_min,
            self._init_value_max,
            self.shape,
        )

    def copy_from(self, source):
        self._table = source._table.copy()

    def randomize(self, rng_generator, std: float = 1.0):
        self._table += rng_generator.normal(*self._table.shape) * std


class CountTable(Table):
    """
    The table counts input occurrences.
    Accepts one or many observations; duplicate indices increment correctly.
    """

    def update(self, *args, **kwargs):
        np.add.at(self._table, tuple(np.stack(args).astype(int)), 1)

    def reset(self, seed=None):
        self._table = np.zeros(self.shape, dtype=np.int32)


class RunningMeanTable(Table):
    """
    The table stores the running mean of an observed variable (such as the reward).
    It expects to receive one new observation at a time.
    """

    def __init__(self, *shape, **kwargs):
        Table.__init__(self, *shape, **kwargs)
        self._count = np.zeros(self.shape)

    def update(self, *args, target=None, **kwargs):
        input = tuple(np.stack(args))
        self._count[input] += 1
        n = self._count[input]
        old_mean = self._table[input]
        new_mean = old_mean * (n - 1) / n + target / n
        self._table[input] = new_mean

    def reset(self, seed=None):
        Table.reset(self, seed=seed)
        self._count.fill(0.0)

    def copy_from(self, source):
        Table.copy_from(self, source)
        self._count = source._count.copy()


class MSETable(Table):
    """
    It is updated using the MSE between the current table value (such as a Q-value)
    and a target (such as the TD target).
    """

    def update(self, *args, target=None, stepsize=1.0, **kwargs):
        """
        If there is more than one sample for the same input pair, the
        gradient is averaged. For example,

            >>> state = np.array([0, 0, 0])
            >>> action = np.array([2, 1, 2])
            >>> target = np.array([0.5, 0.2, 1.0])

        There are two samples for the state-action pair (0, 2), with targets
        0.5 and 1.0, respectively. The Q-value corresponding to this pair will
        be updated using the average of the gradient of those two samples.
        """

        input = np.stack(args)
        prediction = self._table[tuple(input)]
        loss = 0.5 * (target - prediction) ** 2
        gradient = target - prediction
        gradient *= stepsize

        # accumulate gradients per unique key
        unique_input, inv = np.unique(input, axis=1, return_inverse=True)
        m = unique_input.shape[1]
        acc = np.zeros((m,) + gradient.shape[1:], dtype=gradient.dtype)
        np.add.at(acc, inv, gradient)

        counts = np.bincount(inv, minlength=m).astype(acc.dtype)
        counts = counts.reshape((m,) + (1,) * (acc.ndim - 1))
        avg_gradient = acc / counts
        self._table[tuple(unique_input)] += avg_gradient

        return loss, np.linalg.norm(gradient)


# ------------------------------------------------------------------------------
# --------------------------------- ENCODERS -----------------------------------
# ------------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    """Stacks Conv2d + GroupNorm + NonLinear blocks as encoder for image observations.
    Expects state_shape = (C, H, W). Layer geometry is fully
    specified by parallel lists in yaml; `conv_paddings` defaults to zeros,
    `conv_groups` defaults to max(1, channels // 8) per layer, and setting
    `adaptive_pool: True` inserts an AdaptiveAvgPool2d((1, 1)) before Flatten
    (useful for small spatial inputs)."""

    def __init__(
        self,
        state_shape,
        conv_channels,
        conv_kernels,
        conv_strides,
        conv_paddings = None,
        conv_groups = None,
        adaptive_pool = False,
        nonlinearity = "ReLU",
        bias = False,
        **kwargs,
    ):
        super().__init__()
        self.state_shape = state_shape
        self.conv_channels = list(conv_channels)
        self.conv_kernels = list(conv_kernels)
        self.conv_strides = list(conv_strides)
        self.conv_paddings = list(conv_paddings) if conv_paddings is not None else [0] * len(conv_channels)
        self.conv_groups = (
            list(conv_groups) if conv_groups is not None
            else [max(1, c // 8) for c in self.conv_channels]
        )
        self.adaptive_pool = adaptive_pool
        self.nonlinearity = get_layer(nonlinearity)

        layers = [FlattenBatch(self.state_shape)]
        in_ch = self.state_shape[-3]
        for out_ch, ks, st, pad, gr in zip(
            self.conv_channels, self.conv_kernels, self.conv_strides,
            self.conv_paddings, self.conv_groups,
        ):
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=st, padding=pad, bias=bias).apply(zero_init),
                nn.GroupNorm(num_groups=gr, num_channels=out_ch),
                self.nonlinearity(),
            ]
            in_ch = out_ch
        if self.adaptive_pool:
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class RawGridEncoder(nn.Module):
    """Flatten → Gaussian or Fuzzy Tiling."""

    def __init__(self, state_shape, grid_id, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            FlattenBatch(state_shape),
            get_layer(grid_id)(int(np.prod(state_shape)), **kwargs),
        )

    def forward(self, x):
        return self.encoder(x)


class MaxoutGridEncoder(nn.Module):
    """Flatten → LinearN → Max → Gaussian or Fuzzy Tiling."""

    def __init__(
        self,
        state_shape,
        grid_id,
        hidden_size,
        num_layers,
        normalization = None,
        bias = False,
        init = zero_init,
        **kwargs,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            FlattenBatch(state_shape),
            LinearN(
                int(np.prod(state_shape)),
                hidden_size,
                num_layers,
                normalization=get_layer(normalization),
                bias=bias,
                init=init,
                **kwargs,
            ),
            Aggregation(["max"]),
            get_layer(grid_id)(hidden_size, **kwargs),
        )

    def forward(self, x):
        return self.encoder(x)


class GridAggregationEncoder(nn.Module):
    """Flatten → LinearN → Gaussian or Fuzzy Tiling → Aggregation.
    That is, multiple grid-like activations are computed in parallel and independently,
    and then aggregated.

    Note: computation can be expensive with large num_layers and num_centers."""

    def __init__(
        self,
        state_shape,
        grid_id,
        aggregation_stats,
        hidden_size,
        num_layers,
        normalization = None,
        bias = False,
        init = zero_init,
        **kwargs,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            FlattenBatch(state_shape),
            LinearN(
                int(np.prod(state_shape)),
                hidden_size,
                num_layers,
                normalization=get_layer(normalization),
                bias=bias,
                init=init,
                **kwargs,
            ),
            get_layer(grid_id)(hidden_size, **kwargs),
            Aggregation(aggregation_stats),
        )

    def forward(self, x):
        return self.encoder(x)


class PassthroughEncoder(nn.Module):
    """Wraps an encoder for flat observations: features at `passthrough` indices
    (e.g., the binary leg contacts of LunarLander) skip the encoder, and are
    concatenated raw to its output. All other features go through `encoder_cls`.
    Shape: (..., state_dim) → (batch, encoder_dim + num_passthrough)."""

    def __init__(self, encoder_cls, state_shape, passthrough, **kwargs):
        super().__init__()
        assert len(state_shape) == 1, "passthrough requires a flat state_shape"
        self.state_dim = state_shape[0]
        mask = np.zeros(self.state_dim, dtype=bool)
        mask[list(passthrough)] = True
        # Python lists, not buffers: soft_reset blends buffers as floats.
        self.raw_idx = np.flatnonzero(mask).tolist()
        self.enc_idx = np.flatnonzero(~mask).tolist()
        self.encoder = encoder_cls((len(self.enc_idx),), **kwargs)

    def forward(self, x):
        x = x.reshape(-1, self.state_dim)
        return torch.cat([self.encoder(x[:, self.enc_idx]), x[:, self.raw_idx]], dim=-1)


# ------------------------------------------------------------------------------
# ------------------------------ BODY NETWORKS ---------------------------------
# ------------------------------------------------------------------------------

class MSENetwork(FunctionApproximator, nn.Module):
    """
    Its input are states and its output are Q-values for all actions.

        >>> q_fun = MSENetwork(
            state_shape=(9,),
            n_act=4,
            hidden_size=64,
            normalization="LayerNorm",
            nonlinearity="ReLU",
            lrate=0.001,
            optimizer_id="Adam",
            loss="mse_loss",
            max_grad_norm=1.0,
            device="cpu",
            seed=42,
        )
        >>> state = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0])
        >>> values = q_fun(state)
        >>> values.shape
        (4,)
        >>> action = np.array(0)
        >>> values = q_fun(state, action)
        >>> values.shape
        ()
        >>> values = q_fun(np.stack((state, state)))
        >>> values.shape
        (2, 4)
        >>> values = q_fun(np.stack((state, state)), np.stack((action, action)))
        >>> values.shape
        (2,)
    """

    def __init__(
        self,
        state_shape: tuple,
        n_act: int,
        hidden_size: int,
        normalization: str,
        nonlinearity: str,
        lrate: float,
        optimizer_id: str,
        loss: str,
        bias: bool,
        maxout_layers: int,
        dropout_p: float,
        max_grad_norm: float = None,
        device: str = "cpu",
        encoder = None,
        seed: int = None,
        **kwargs,
    ):
        nn.Module.__init__(self)
        FunctionApproximator.__init__(self)

        self.state_shape = state_shape
        self.n_output = n_act
        self._loss = getattr(nn.functional, loss)
        self._optimizer_id = optimizer_id
        self._lrate = lrate
        self._hidden_size = hidden_size
        self._maxout_layers = maxout_layers
        self._bias = bias
        self._dropout_p = dropout_p
        self.Linear = get_layer("Linear")
        self.Norm = get_layer(normalization)
        self.NonLinear = get_layer(nonlinearity)
        self._max_grad_norm = max_grad_norm
        self.device = get_torch_device(device)
        self._encoder_cfg = encoder
        self.to(self.device)
        self.reset(seed=seed)

    def to(self, device):  # Override `to` to keep track of the device
        self.device = torch.device(device)
        return super().to(device)

    # Do not override `__call__` since all hooks are associated to it.
    # Override `forward` and implement private `_forward` functions depending on
    # the network structure.

    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    def _forward(self, state):
        h = self._obs_encoder(state)
        h = self._body(h)
        return self._action_head(h)

    def forward(self, state, action=None, with_gradient=False, **kwargs):
        # Batch dimensions are always first, e.g., (batch, time sequence, features)
        # or (batch, time sequence, channels, height, width).
        # Then, action dimension comes next in the output, followed by extra dimensions,
        # e.g., (batch, time sequence, action, head).
        batch_shapes = state.shape[:-len(self.state_shape)]
        action_dim = len(batch_shapes)

        with torch.set_grad_enabled(with_gradient):
            q = self._forward(
                torch.asarray(state, dtype=torch.float, device=self.device),
            ).view(*batch_shapes, self.n_output)

        if action is not None:
            action = torch.asarray(action, device=self.device, dtype=torch.long)
            q = torch.take_along_dim(
                q,
                action.unsqueeze(action_dim),
                dim=action_dim,
            ).squeeze(action_dim)

        return q.detach().cpu().numpy() if not with_gradient else q

    def print_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)
                print(param.data)

    def update(self, *args, target=None, stepsize=1.0, **kwargs):
        prediction = self(*args, with_gradient=True)
        loss = self._loss(
            torch.asarray(target, device=self.device, dtype=torch.float),
            prediction,
            reduction="none",
        )
        self._optimizer.zero_grad(set_to_none=True)
        ((loss * torch.asarray(stepsize, device=self.device)).mean()).backward()
        if self._max_grad_norm is None:
            gradient_norm = torch.nn.utils.get_total_norm(self.parameters())
        else:
            gradient_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self._max_grad_norm).item()
        self._optimizer.step()
        return loss.detach().cpu().numpy(), gradient_norm, prediction.detach().cpu().numpy()

    def reset(self, seed=None):
        # rng_generator = self.rng_generator(seed)
        # Currently, PyTorch does not support passing generator to init functions.
        # For example, you cannot do `nn.Linear(..., rng_generator)`.
        # Either you call custom init function (like `zero_init`) with the generator
        # for all layers (but some do not support it), or set the seed globally (as below).
        # Also, functions like Dropout do not support generators.
        if seed is not None:
            torch.manual_seed(seed)

        # Cannot use LazyLinear and also have custom init, unless you
        # make a custom Module and override reset_parameters().

        encoder_id = None if self._encoder_cfg is None else self._encoder_cfg.get('id')

        if encoder_id is None:
            self._obs_encoder = nn.Identity().to(self.device)
        else:
            encoder_kwargs = {k: v for k, v in self._encoder_cfg.items() if k not in ('id', 'passthrough')}
            encoder_cls = getattr(src.approximator, encoder_id)
            passthrough = self._encoder_cfg.get('passthrough')
            if passthrough is None:
                self._obs_encoder = encoder_cls(self.state_shape, **encoder_kwargs)
            else:
                self._obs_encoder = PassthroughEncoder(
                    encoder_cls, self.state_shape, passthrough, **encoder_kwargs,
                )
            self._obs_encoder.to(self.device)

        self._obs_encoder.eval()
        state_enc_dim = self._obs_encoder(
            torch.zeros((1,) + tuple(self.state_shape), device=self.device)
        ).shape[-1]
        self._obs_encoder.train()

        self._body = nn.Sequential(
            self.Norm(state_enc_dim) if encoder_id is not None else nn.Identity(),
            # This is a Maxout block (Goodfellow et al., 2013)
            LinearN(
                state_enc_dim,
                self._hidden_size,
                self._maxout_layers,
                normalization=nn.Identity,
                bias=self._bias,
                init=zero_init,
            ),
            Aggregation(["max"]),
            self.Norm(self._hidden_size),
            self.NonLinear(),
            nn.Dropout(self._dropout_p) if self._dropout_p > 0.0 else nn.Identity(),
        ).to(self.device)

        self._action_head = nn.Sequential(
            self.Linear(self._hidden_size, self._hidden_size, bias=self._bias).apply(zero_init),
            self.Norm(self._hidden_size),
            self.NonLinear(),
            self.Linear(self._hidden_size, self.n_output).apply(zero_init),
        ).to(self.device)

        self.reset_optimizer()

    def reset_optimizer(self):
        self._optimizer = getattr(torch.optim, self._optimizer_id)(
            params=self.parameters(),
            lr=self._lrate,
        )

    def soft_reset(self, alpha: float, seed: int = None):
        src.utils.torch_nn.soft_reset(self, alpha, seed)

    def soft_reset_optimizer(self, alpha: float):
        src.utils.torch_nn.soft_reset_optimizer(self._optimizer, alpha)

    def copy_from(self, source, tau: float = 1.0):
        self._encoder_cfg = source._encoder_cfg
        if tau == 1.0:  # hard copy
            self.load_state_dict(source.state_dict())
        else:  # soft copy
            for t, s in zip(self.parameters(), source.parameters()):
                t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


class DuelingMSENetwork(MSENetwork):
    """MSENetwork with separate advantage and state-value heads, combined into Q-values
    (Wang et al., 2016, https://arxiv.org/abs/1511.06581)."""

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._state_head = nn.Sequential(
            self.Linear(self._hidden_size, self._hidden_size, bias=self._bias).apply(zero_init),
            self.Norm(self._hidden_size),
            self.NonLinear(),
            self.Linear(self._hidden_size, 1).apply(zero_init),
        ).to(self.device)
        self.reset_optimizer()

    def _forward(self, state):
        h = self._obs_encoder(state)
        h = self._body(h)
        return self._action_head(h), self._state_head(h)

    def forward(self, state, action=None, with_gradient=False, **kwargs):
        batch_shapes = state.shape[:-len(self.state_shape)]
        action_dim = len(batch_shapes)

        with torch.set_grad_enabled(with_gradient):
            a, v = self._forward(
                torch.asarray(state, dtype=torch.float, device=self.device),
            )
            a = a.view(*batch_shapes, self.n_output)
            v = v.view(*batch_shapes, 1)
            q = v + (a - a.mean(dim=-1, keepdim=True))

        if action is not None:
            action = torch.asarray(action, device=self.device, dtype=torch.long)
            q = torch.take_along_dim(
                q,
                action.unsqueeze(action_dim),
                dim=action_dim,
            ).squeeze(action_dim)

        return q.detach().cpu().numpy() if not with_gradient else q
