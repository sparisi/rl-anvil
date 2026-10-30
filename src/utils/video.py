"""
Concatenates a list of frames in a grid-like manner.
Useful to concatenate many videos and play them side-to-side.
Frames are concatenated taking their size into account, such that the ratio of
the concatenated frame is as close as possible to 1.

Example:

>>> import gymnasium
>>> import cv2
>>>
>>> n_envs = 20
>>> envs = [
>>>     gymnasium.make("Pendulum-v1", render_mode="rgb_array")
>>>     for _ in range(n_envs)
>>> ]
>>>
>>> for e in envs:
>>>     _ = e.reset()
>>>
>>> frames = []
>>> for t in range(100):
>>>     step_frames = []
>>>     for e in envs:
>>>         _ = e.step(e.action_space.sample())
>>>         step_frames.append(e.render())
>>>     frames.append(frames_to_grid(step_frames))
>>>
>>> out = cv2.VideoWriter(
>>>     "concat_videos.mp4",
>>>     cv2.VideoWriter_fourcc(*'mp4v'),
>>>     30,
>>>     (frames[0].shape[1], frames[0].shape[0]),
>>> )
>>>
>>> for frame in frames:
>>>     out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
>>>
>>> out.release()
"""

import math

import numpy as np


def reshape_frames_into_grid(n_frames, h, w, target_aspect_ratio=1.0):
    # Find every (rows, cols) with rows * cols == n_frames, i.e., every way to
    # arrange the frames.
    grids = [
        (rows, n_frames // rows)
        for rows in range(1, n_frames + 1)
        if n_frames % rows == 0
    ]
    # When the only exact grids are a single row and a single column (n_frames is
    # a prime number), pad the last row with empty cells.
    if len(grids) <= 2:
        grids += [
            (rows, math.ceil(n_frames / rows))
            for rows in range(2, n_frames)
            if (rows - 1) * math.ceil(n_frames / rows) < n_frames
        ]
    best_rows, best_cols = 1, n_frames
    min_diff = np.inf
    for rows, cols in grids:
        total_height = rows * h
        total_width = cols * w
        aspect = total_width / total_height
        diff = abs(math.log(aspect / target_aspect_ratio))
        if diff < min_diff:
            min_diff = diff
            best_rows, best_cols = rows, cols
    return best_rows, best_cols

def frames_to_grid(frames, target_aspect_ratio=1.0):
    frames = np.array(frames)
    n_frames, h, w, c = frames.shape
    rows, cols = reshape_frames_into_grid(n_frames, h, w, target_aspect_ratio)
    grid = np.zeros((rows * h, cols * w, c), dtype=frames.dtype)
    for idx in range(n_frames):
        r = idx // cols
        c_ = idx % cols
        grid[r * h : (r + 1) * h, c_ * w : (c_ + 1) * w] = frames[idx]
    return grid
