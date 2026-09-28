"""Merge per-checkpoint media into one animation per run.

A run launched with `results.save_heatmaps` writes one PNG per checkpoint to
`<data_dir>/<config_id>/<seed>/heatmaps/<steps>.png`, and one launched with
`results.save_videos` writes one clip per checkpoint to
`<data_dir>/<config_id>/<seed>/videos/<steps>.{mp4,gif}`.
This script walks a data directory and, for every such folder, writes a single
merged animation next to it:

    <seed>/heatmaps/  ->  <seed>/heatmaps.{mp4,gif}
    <seed>/videos/    ->  <seed>/videos.{mp4,gif}

Every frame is captioned "Steps X" on a band across the top, where X is the
checkpoint the frame came from, i.e. the source file name. A video clip
contributes many frames, all captioned with that clip's checkpoint.

Without --gif or --mp4, the output follows the source: mp4 clips merge into an
mp4, gif clips into a gif, and a folder holding both produces one output per
type. Heatmap frames are PNGs and carry no source format, so they default to mp4.
--gif and --mp4 force that format for heatmaps and videos alike, and clips of
both source formats then merge into the single forced output. They cannot be
passed together. Source frame rate is preserved unless --fps overrides it.

`--checkpoints N` thins the videos down to N checkpoints, equally spaced and
always spanning first to last, which is what turns a run with hundreds of clips
into a short animation. It defaults to 10, must be at least 3, and accepts `None`
to keep every clip. Heatmaps ignore it and always keep every checkpoint: a
heatmap checkpoint is one frame, so even hundreds of them merge into a short
animation, whereas a video checkpoint is a whole clip and a handful of them
may already result in a large video.

--fps is the playback rate, and both formats run at it. An mp4 is given it
directly. A GIF cannot be: it has no frame rate, only a per-frame delay in whole
hundredths of a second, so the rates it can PLAY at are 100/n -- 50, 33.3, 25, 20
-- and a request it cannot play is reached by playing slower and keeping fewer
frames. 60 fps is 20 fps showing one frame in three, which crosses the source at
60 frames a second exactly as the mp4 does. For heatmaps that means dropping
checkpoints, which is what a GIF costs above 50 fps.

--frameskip_videos drops frames from the clips: 1 keeps every other frame, 2 one
in three. The animation is that many times faster and that many times smaller,
where --fps changes the speed without changing what is in the file.

--resize_heatmaps and --resize_videos scale the frames before they are merged --
0.4 for 40% of the original size -- which is the lever on file size, where --fps
is the lever on how long the animation runs. Neither goes above 1.

Frames are streamed: the output canvas is sized from file headers first, then
each frame is decoded, labelled, written, and dropped. Only one frame is held at
a time, so a folder of long clips costs no more memory than a folder of short
ones.

Usage:
    python merge_media.py -f data_demo
    python merge_media.py -f data_demo/1a95d2bb --gif --fps 4
    python merge_media.py -f data_demo/1a95d2bb/0 -n none -v
"""

import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import imageio.v2 as imageio  # MP4 encoding/decoding via imageio-ffmpeg

HEATMAP_SUFFIXES = {".png"}
VIDEO_SUFFIXES = {".mp4", ".gif"}
DEFAULT_FPS = 5.0  # only used when the source carries no frame rate
FONT_CANDIDATES = ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf")


def n_checkpoints(value):
    """Argparse type: an int >= 3, or `none` to keep every clip."""
    if str(value).lower() in ("none", "all"):
        return None
    n = int(value)
    if n < 3:
        raise argparse.ArgumentTypeError(f"must be at least 3 (first, mid, last), got {n}")
    return n


def scale_factor(value):
    """Argparse type: a resize factor in (0, 1], where 0.4 means 40% of the
    original size and 1 leaves the frames alone.

    Above 1 is refused. These animations are read at a glance in a README, and
    enlarging a frame past what was rendered adds bytes without adding detail --
    the way to a bigger picture is a bigger figure, not a scaled-up one."""

    factor = float(value)
    if not 0 < factor <= 1:
        raise argparse.ArgumentTypeError(
            f"must be greater than 0 and at most 1, got {value}")
    return factor


def frames_to_skip(value):
    """Argparse type: how many frames to drop after each one kept, 0 or more."""

    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"cannot be negative, got {value}")
    return n


parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("-f", "--folder", default="data",
                    help="Data directory to scan recursively.")
parser.add_argument("-n", "--checkpoints", type=n_checkpoints, default=10,
                    help="How many video checkpoints to show, equally spaced from first "
                         "to last. At least 3, or None to keep them all. Default: 10. "
                         "Heatmaps always keep every checkpoint.")
format_group = parser.add_mutually_exclusive_group()
format_group.add_argument("--gif", action="store_true",
                          help="Force .gif output for heatmaps and videos.")
format_group.add_argument("--mp4", action="store_true",
                          help="Force .mp4 output for heatmaps and videos.")
parser.add_argument("--frameskip_videos", type=frames_to_skip, default=0, metavar="N",
                    help="Drop N frames after every frame kept: 1 keeps every other "
                         "frame, 2 keeps one in three. The animation runs the same "
                         "length at N times the speed, and the file shrinks with the "
                         "frames. Videos only. Default: 0, every frame.")
parser.add_argument("--resize_heatmaps", type=scale_factor, default=None, metavar="F",
                    help="Scale heatmap frames by F before merging, e.g. 0.4 for 40%% "
                         "of their original size. Default: unchanged.")
parser.add_argument("--resize_videos", type=scale_factor, default=None, metavar="F",
                    help="Scale video frames by F before merging, e.g. 0.4 for 40%% "
                         "of their original size. Default: unchanged.")
parser.add_argument("--fps", type=float, default=None,
                    help="Playback frame rate. Default: 5 for heatmaps, and the "
                         "source rate for videos.")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Show a progress bar over the files being read.")
args = parser.parse_args()


def forced_suffix():
    """Output suffix demanded by --gif/--mp4, or None to follow the source."""
    if args.gif:
        return ".gif"
    if args.mp4:
        return ".mp4"
    return None


_font_cache = {}


def get_font(width):
    """(font, size) scaled to the frame width, falling back to Pillow's default."""
    size = max(12, width // 30)
    if size not in _font_cache:
        font = None
        for name in FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=size)  # Pillow >= 10.1
            except TypeError:
                font = ImageFont.load_default()
        _font_cache[size] = (font, size)
    return _font_cache[size]


def band_height(width):
    """Height of the caption band added above a frame of the given width."""
    _, size = get_font(width)
    return int(round(size * 1.8))


def resized(image, factor):
    """`image` scaled by `factor`, or `image` itself when there is nothing to do.

    Applied before the caption is drawn, so the band is sized from the width the
    frame ends up at and the text stays proportionate rather than shrinking with
    the picture."""

    if factor is None or factor == 1:
        return image
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    return image.resize(size, Image.LANCZOS)


def label_frame(image, step):
    """Copy of `image` with "Steps <step>" on a band added above it."""
    font, _ = get_font(image.width)
    band = band_height(image.width)
    labelled = Image.new("RGB", (image.width, image.height + band), (255, 255, 255))
    labelled.paste(image, (0, band))

    text = f"Steps {step}"
    draw = ImageDraw.Draw(labelled)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((image.width - (right - left)) // 2 - left, (band - (bottom - top)) // 2 - top),
        text,
        fill=(0, 0, 0),
        font=font,
    )
    return labelled


def step_files(folder, suffixes):
    """(step, path) pairs for files named after an integer step, sorted by step.

    The integer-name requirement is what distinguishes a checkpoint folder from
    any other folder that happens to be called `videos` or `heatmaps`.
    """
    found = []
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            step = int(path.stem)
        except ValueError:
            continue
        found.append((step, path))
    return sorted(found)


def select_checkpoints(items):
    """Equally spaced subset of (step, path) pairs, spanning first to last.

    `np.linspace` over the indices puts the endpoints on the first and last
    checkpoint, so the animation always covers the whole run.

    Videos only. Each clip here contributes many frames, so keeping them all
    makes an animation nobody watches to the end; a heatmap checkpoint is one
    frame and they are all kept.
    """
    n = args.checkpoints
    if n is None or len(items) <= n:
        return items
    return [items[i] for i in np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))]


def source_size(path):
    """(width, height) of a source file, read from its header rather than decoded.

    This is what lets the canvas be sized without holding any frames.
    """
    if path.suffix.lower() in HEATMAP_SUFFIXES:
        with Image.open(path) as image:
            return image.size
    with imageio.get_reader(path) as reader:
        size = reader.get_meta_data().get("size")
        if size is not None:
            return int(size[0]), int(size[1])
        first = np.asarray(reader.get_data(0))  # gif readers may not report a size
        return first.shape[1], first.shape[0]


def source_fps(path):
    """Frame rate declared by an encoded clip, or None."""
    with imageio.get_reader(path) as reader:
        meta = reader.get_meta_data()
    fps = meta.get("fps")
    if fps is None:
        duration = meta.get("duration")  # gif: milliseconds per frame
        fps = 1000.0 / duration if duration else None
    return fps


def canvas_size(items, is_gif, factor=None):
    """Output canvas: large enough for every labelled frame in `items`, at the
    size `factor` scales them to.

    Figures saved with bbox_inches="tight" vary in size between checkpoints, so
    the animation needs one canvas that fits the largest of them.
    """
    width = height = 0
    for _, path in items:
        w, h = source_size(path)
        if factor is not None and factor != 1:
            w, h = max(1, round(w * factor)), max(1, round(h * factor))
        width = max(width, w)
        height = max(height, h + band_height(w))
    if not is_gif:  # libx264 needs even dimensions
        width += width % 2
        height += height % 2
    return width, height


def place(frame, canvas):
    """`frame` centered on a white canvas of the given size, as a numpy array."""
    if frame.size != canvas:
        padded = Image.new("RGB", canvas, (255, 255, 255))
        padded.paste(frame, ((canvas[0] - frame.width) // 2, (canvas[1] - frame.height) // 2))
        frame = padded
    return np.asarray(frame)


_reported_rates = set()


def gif_plan(fps):
    """How a GIF runs at `fps`, as (centiseconds per frame, source frames per
    written frame).

    A GIF has no frame rate. Each frame carries a delay in whole hundredths of a
    second, so the only rates it can PLAY at are 100/n: 50, 33.3, 25, 20 and so
    on, and a delay of one centisecond has meant "unspecified" since the format
    was written -- every viewer answers it with 100 ms, which is why asking for
    60 fps and getting 10 is possible.

    So a rate it cannot play is reached by playing slower and showing fewer
    frames: 60 fps is 20 fps keeping every third frame, and the animation crosses
    the source at the rate that was asked for. The pair chosen is the one whose
    product lands closest to `fps`, preferring the smoothest playback among ties.
    """

    best = None
    # 2 centiseconds is the fastest a viewer honours, 1000 the slowest anyone
    # would sit through: 50 fps down to one frame every ten seconds.
    for cs in range(2, 1001):
        keep = max(1, round(fps * cs / 100))  # source frames per written frame
        error = abs(100 / cs * keep - fps)
        # Ties go to the smaller delay, which keeps more of the frames: `keep`
        # grows with `cs`, so the smoothest plan of equal accuracy wins.
        if best is None or (error, cs) < best[0]:
            best = ((error, cs), (cs, keep))
    return best[1]


def playable_fps(fps):
    """The rate both formats can play at, as (fps, centiseconds per frame).

    A GIF has no frame rate. Each frame carries a delay in whole hundredths of a
    second, so the only rates it can express are 100/n: 50, 33.3, 25, 20 and so
    on. A request between them is truncated, and a delay of one centisecond has
    meant "unspecified" since the format was written, which every viewer answers
    with 100 ms -- so asking a GIF for 60 fps gets 10.

    An mp4 carries a real timebase and would honour the request exactly, which is
    how the same --fps ends up playing at two speeds. Both are given the rounded
    rate instead: one flag, one speed, whatever the output format."""

    cs, keep = gif_plan(fps)
    speed = 100 / cs * keep
    if keep > 1 and fps not in _reported_rates:
        _reported_rates.add(fps)
        tqdm.write(f"{fps:g} fps in a GIF: playing at {100 / cs:g} fps and keeping "
                   f"1 frame in {keep}, which runs at {speed:g} fps -- a GIF can only "
                   f"play at 100/n fps, so a faster one is reached by showing fewer "
                   f"frames rather than by slowing down.")
    elif abs(speed - fps) > 1e-9 and fps not in _reported_rates:
        _reported_rates.add(fps)
        tqdm.write(f"Frame rate {fps:g} rounded to {speed:g} fps ({cs} cs per frame): "
                   f"a GIF can only play at 100/n fps.")
    return cs, keep


def open_writer(path, fps):
    """Streaming writer for .gif or .mp4, and how many source frames each written
    frame stands for -- always 1 for an mp4, which plays whatever rate it is
    given.

    The mp4 gets the rate that was asked for; the GIF gets the delay and the
    frame skip that run at that same rate (see gif_plan). The delay is written in
    milliseconds rather than as a rate, since a rate has to be turned back into
    one and 100/3 fps comes back as 29.999 ms, which truncates to 2 centiseconds
    instead of 3."""

    if path.suffix.lower() == ".gif":
        cs, keep = playable_fps(fps)
        return imageio.get_writer(path, mode="I", duration=cs * 10, loop=0), keep
    return imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8, macro_block_size=1,
    ), 1


def merge_heatmaps(folder, items, pbar):
    """Merge the given <step>.png frames into one animation beside `folder`."""
    # PNG frames carry no source format, so .mp4 is the default to follow.
    out = folder.with_suffix(forced_suffix() or ".mp4")
    canvas = canvas_size(items, out.suffix == ".gif", args.resize_heatmaps)
    writer, keep = open_writer(out, args.fps or DEFAULT_FPS)
    written = 0
    with writer:
        for i, (step, path) in enumerate(items):
            # A heatmap frame is a whole checkpoint, so skipping one drops a
            # checkpoint from the animation rather than a repeated picture. That
            # is what a GIF costs at a rate it cannot play; the last frame is kept
            # whatever the skip lands on, so the animation still ends where the
            # run did.
            if i % keep and i != len(items) - 1:
                pbar.update(1)
                continue
            with Image.open(path) as image:
                frame = label_frame(
                    resized(image.convert("RGB"), args.resize_heatmaps), step)
            writer.append_data(place(frame, canvas))
            written += 1
            pbar.update(1)
    tqdm.write(f"{out}  ({written} frames from {len(items)} checkpoints)")
    return 1


def merge_videos(folder, groups, pbar):
    """Concatenate the given clips into one animation per output format."""
    written = 0
    for suffix, items in sorted(groups.items()):
        out = folder.with_suffix(suffix)
        canvas = canvas_size(items, suffix == ".gif", args.resize_videos)
        fps = args.fps or source_fps(items[0][1]) or DEFAULT_FPS
        n_frames = 0
        writer, keep = open_writer(out, fps)
        with writer:
            for step, path in items:
                with imageio.get_reader(path) as reader:
                    # Counted across the whole clip, not restarted per file: the
                    # skip is what makes a GIF run at a rate it cannot play, and
                    # restarting it would stall on every clip boundary.
                    #
                    # Two skips, multiplied rather than applied in turn: the GIF's
                    # is how it reaches a rate it cannot play, --frameskip_videos
                    # is a rate the reader asked to go faster at. Keeping one
                    # frame in three of every other frame is keeping one in six.
                    for i, raw in enumerate(reader):
                        if i % (keep * (args.frameskip_videos + 1)):
                            continue
                        frame = label_frame(
                            resized(Image.fromarray(np.asarray(raw)).convert("RGB"),
                                    args.resize_videos),
                            step,
                        )
                        writer.append_data(place(frame, canvas))
                        n_frames += 1
                pbar.update(1)
        tqdm.write(f"{out}  ({n_frames} frames from {len(items)} clips)")
        written += 1
    return written


root = Path(args.folder)
if not root.is_dir():
    raise SystemExit(f"Not a directory: {root}")

# Resolve which files each folder contributes up front, so the bars below can be
# sized in files rather than ticking once per folder.
heatmap_jobs, video_jobs = [], []
for folder in sorted(root.rglob("*")):
    if not folder.is_dir():
        continue
    if folder.name == "heatmaps":
        items = step_files(folder, HEATMAP_SUFFIXES)
        if items:
            heatmap_jobs.append((folder, items))
    elif folder.name == "videos":
        items = step_files(folder, VIDEO_SUFFIXES)
        if forced_suffix() is not None:
            # One output, so clips of both source formats cannot collide on it.
            groups = {forced_suffix(): select_checkpoints(items)} if items else {}
        else:
            by_suffix = defaultdict(list)
            for step, path in items:
                by_suffix[path.suffix.lower()].append((step, path))
            groups = {s: select_checkpoints(group) for s, group in by_suffix.items()}
        if groups:
            video_jobs.append((folder, groups))

merged = 0

if heatmap_jobs:
    total = sum(len(items) for _, items in heatmap_jobs)
    with tqdm(total=total, desc="Heatmaps", unit="png", disable=not args.verbose) as pbar:
        for folder, items in heatmap_jobs:
            # The run folder, not the media folder -- the media folder's name is
            # already the bar's description.
            pbar.set_postfix_str(str(folder.parent.relative_to(root)))
            merged += merge_heatmaps(folder, items, pbar)

if video_jobs:
    total = sum(len(items) for _, groups in video_jobs for items in groups.values())
    with tqdm(total=total, desc="Videos", unit="clip", disable=not args.verbose) as pbar:
        for folder, groups in video_jobs:
            pbar.set_postfix_str(str(folder.parent.relative_to(root)))
            merged += merge_videos(folder, groups, pbar)

print(f"\nDone, {merged} animation(s) written.")
