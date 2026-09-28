from abc import ABC, abstractmethod


class Parameter(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def step(self, **kwargs):
        pass

    def copy_from(self, source):
        if type(self) is not type(source):
            raise TypeError(f"cannot copy {type(source)} into {type(self)}")
        self.__dict__.update(source.__dict__)  # All attributes are float or int

    @property
    def value(self):
        return self._value

    def reset(self):
        self._t = 0
        self._value = self._init_value


class Constant(Parameter):
    """
    Constant parameter.

    Args:
        value (float): the value of the parameter,
    """

    def __init__(self, value: float, **kwargs):
        self._init_value = value
        self._t = 0
        self._value = value

    def step(self):
        self._t += 1


class LinearDecay(Parameter):
    """
    Parameter decaying linearly at a given 'rate'.

        k_t = clip(k_0 - max(t - constant_init_steps, 0) * rate, end_value)

    If the end value is larger than the init value or if the rate is negative,
    the parameter will not decay but grow instead.

    'rate' can be either passed explicitly or derived given 'steps', 'constant_init_steps'
    (how many steps before decay starts) and 'constant_end_steps' (how many steps
    after decay ends).

    Example:

        LinearDecay(
            init_value=1.0,
            end_value=0.1,
            steps=100,
            constant_init_steps=10,
            constant_end_steps=10,
        )

        Then rate = (1.0 - 0.1) / (100 - 10 - 10) = 0.01125.
        That is, the value stays at 1.0 for the first 10 steps, decays linearly
        from 1.0 to 0.1 over the next 80 steps, then stays at 0.1 for the last 10 steps.

    Note:
        If 'rate' is passed explicitly, its sign must match the direction from
        init_value to end_value: positive rate decays down (init > end),
        negative rate grows up (init < end). A mismatched sign will drive the
        value AWAY from end_value forever, and the end_value clip will never
        engage. When 'steps' is used instead, the rate is derived automatically
        and always has the correct sign.

    Args:
        init_value (float): the initial value of k,
        end_value (float): the final value of k,
        constant_init_steps (int, 0): how many steps before decaying starts,
        constant_end_steps (int, 0): how many steps before 'steps' the value
            should already have reached end_value, i.e. how long the value
            stays constant at the end,
        rate (float, None): the decay rate,
        steps (int, None): if the rate is not defined, it will be calculated
            automatically as
            rate = (init_value - end_value) / (steps - constant_init_steps - constant_end_steps),
    """

    def __init__(
        self,
        init_value: float,
        end_value: float = None,
        constant_init_steps: int = 0,
        constant_end_steps: int = 0,
        rate: float = None,
        steps: int = None,
        **kwargs,
    ):
        assert constant_init_steps >= 0, (
            f"'constant_init_steps' must be non-negative (received {constant_init_steps})"
        )
        assert constant_end_steps >= 0, (
            f"'constant_end_steps' must be non-negative (received {constant_end_steps})"
        )

        self._init_value = init_value
        self._end_value = end_value
        self._t = 0
        self._t_warm = constant_init_steps

        if steps is not None and end_value is not None:
            decay_steps = steps - constant_init_steps - constant_end_steps
            self._rate = (init_value - end_value) / max(decay_steps, 1)
        else:
            if rate is not None:
                self._rate = rate
            else:
                assert init_value == end_value, (
                    f"to decay from {init_value} to {end_value} "
                    f"you must pass either 'rate' or 'steps'"
                )
                self._rate = 0.0
        self._value = init_value

    def step(self):
        self._t += 1
        if self._t <= self._t_warm:  # still in the constant initial phase
            return
        self._value = self._value - self._rate
        if self._end_value is not None:
            if self._rate >= 0.0:
                self._value = max(self._value, self._end_value)
            else:
                self._value = min(self._value, self._end_value)


class ExponentialDecay(Parameter):
    """
    Parameter decaying exponentially at a given 'rate'.

        k_t = clip(k_0 * rate ** max(t - constant_init_steps, 0), end_value)

    If the end value is larger than the init value or if the rate is larger than
    1.0, the parameter will not decay but grow instead.

    'rate' can be either passed explicitly or derived as in LinearDecay.

    Args:
        init_value (float): the initial value of k,
        end_value (float): the final value of k,
        constant_init_steps (int, 0): how many steps before decaying starts,
        constant_end_steps (int, 0): how many steps before 'steps' the value
            should already have reached end_value, i.e. how long the value
            stays constant at the end,
        rate (float, None): the decay rate,
        steps (int, None): if the rate is not defined, it will be calculated
            automatically as
            rate = (end_value / init_value) ** (1.0 / (steps - constant_init_steps - constant_end_steps)),
    """

    def __init__(
        self,
        init_value: float,
        end_value: float = None,
        constant_init_steps: int = 0,
        constant_end_steps: int = 0,
        rate: float = None,
        steps: int = None,
        **kwargs,
    ):
        assert constant_init_steps >= 0, (
            f"'constant_init_steps' must be non-negative (received {constant_init_steps})"
        )
        assert constant_end_steps >= 0, (
            f"'constant_end_steps' must be non-negative (received {constant_end_steps})"
        )

        if end_value is not None:
            assert 1e-6 <= end_value <= 1e6, (
                f"'end_value' must be in [1e-6, 1e6] for numerical stability; got {end_value}"
            )

        self._init_value = init_value
        self._end_value = end_value
        self._t = 0
        self._t_warm = constant_init_steps

        if steps is not None and end_value is not None:
            decay_steps = steps - constant_init_steps - constant_end_steps
            self._rate = (end_value / init_value) ** (1.0 / max(decay_steps, 1))
        else:
            if rate is not None:
                self._rate = rate
            else:
                assert init_value == end_value, (
                    f"to decay from {init_value} to {end_value} "
                    f"you must pass either 'rate' or 'steps'"
                )
                self._rate = 1.0
        self._value = init_value

    def step(self):
        self._t += 1
        t = max(self._t - self._t_warm, 0)  # constant during the initial phase
        self._value = min(max(self._init_value * self._rate**t, 1e-6), 1e6)
        if self._end_value is not None:
            if self._rate >= 1.0:
                self._value = min(self._value, self._end_value)
            else:
                self._value = max(self._value, self._end_value)
