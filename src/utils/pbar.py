from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)
from rich.table import Table

from src.utils.misc import format_value


class MultiProgressBar:
    """
    Class to handle multiple progress bars running in parallel, with an optional
    list of statistics underneath them.
    Useful to print the progress of (for example) training and testing when they
    are running in parallel.

    The bars and the statistics share one live display, so set_stats() overwrites
    the previous statistics in place rather than scrolling a new row past the
    bars. Statistics are listed one per line, so adding statistics makes the
    block taller instead of breaking lines. Only the latest statistics are on
    screen: the history lives in the npz (and in W&B).

    Example:
    >>> import threading
    >>> import time
    >>> from src.utils.pbar import MultiProgressBar
    ...
    >>> def worker(pbar, bar_id, steps, delay):
    ...     for i in range(steps + 1):
    ...     pbar.update(bar_id, i)
    ...     pbar.set_stats({"loss": 0.1234, "updates": i, "clipped": True})
    ...     time.sleep(delay)
    ...
    >>> mpb = MultiProgressBar()
    >>> mpb.add_task("train", 50)
    >>> mpb.add_task("test", 20)
    ...
    >>> def train_thread():
    ...     worker(mpb, "train", 50, 0.05)
    ...
    >>> def test_thread():
    ...     worker(mpb, "test", 20, 0.12)
    ...
    >>> t1 = threading.Thread(target=train_thread)
    >>> t2 = threading.Thread(target=test_thread)

    >>> with mpb:
    ...     t1.start()
    ...     t2.start()
    ...     t1.join()
    ...     t2.join()

    ⠸ train ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━━━  82% • step 41/50 0:00:07 0:00:01
    ⠸ test  ━━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  20% • step 10/50 0:00:03 0:00:01
       loss  0.1234
    updates      41
    clipped     True
    """

    def __init__(
        self,
        console: Console = None,
        key_style: str = "bright_magenta",
    ):
        """
        Args:
            console (Console): (optional) rich Console,
            key_style (str): rich style for the statistics keys.
        """

        self.console = console or Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            "step",
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            refresh_per_second=5,
        )
        self.tasks = {}
        self._key_style = key_style
        self._stats = None
        # Progress is never started, only rendered: starting it would open a
        # second live display on the same console, and two of them fight over
        # the cursor. This one drives both the bars and the statistics.
        self.live = Live(
            console=self.console,
            get_renderable=self._renderable,
            refresh_per_second=5,
            transient=False,
        )

    def _renderable(self) -> RenderableType:
        bars = self.progress.get_renderable()
        if self._stats is None:
            return bars
        return Group(bars, self._stats)

    def add_task(self, id, steps):
        if id in self.tasks:
            raise ValueError(f"a progress bar with the name {id} already exists")
        self.tasks[id] = self.progress.add_task(id, total=steps)

    def __enter__(self):
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.live.__exit__(exc_type, exc_val, exc_tb)

    def update(self, id, completed):
        self.progress.update(self.tasks[id], completed=completed)

    def set_stats(self, stats: dict):
        """
        Replace the statistics shown under the bars, one key/value pair per line.
        """

        if not stats:
            self._stats = None
        else:
            grid = Table.grid(padding=(0, 2))
            grid.add_column(justify="right", style=self._key_style)
            grid.add_column(justify="right")
            for k, v in stats.items():
                grid.add_row(k, format_value(v))
            self._stats = grid
        self.refresh()

    def refresh(self):
        self.live.refresh()
