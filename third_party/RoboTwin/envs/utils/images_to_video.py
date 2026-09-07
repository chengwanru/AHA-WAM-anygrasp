import cv2
import numpy as np
import os
import shutil
import subprocess
import pickle
import pdb
from pathlib import Path


def resolve_ffmpeg_exe() -> str:
    """Resolve a usable ffmpeg binary (RoboTwin calls bare 'ffmpeg')."""
    candidates = []
    which = shutil.which("ffmpeg")
    if which:
        candidates.append(which)
    for root in (
        Path("/opt/huawei/dataset/cwr_wulan_aha"),
        Path("/home/ma-user/work/dataset/cwr_wulan_aha"),
    ):
        candidates.append(str(root / "bin" / "ffmpeg"))
        candidates.append(str(root / "ffmpeg-bin" / "ffmpeg"))
        candidates.append(str(root / "ffmpeg-bin" / "ffmpeg-linux-x86_64-v7.0.2"))
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        "ffmpeg not found. Expected cwr_wulan_aha/bin/ffmpeg or imageio_ffmpeg binary."
    )


def images_to_video(imgs: np.ndarray, out_path: str, fps: float = 30.0, is_rgb: bool = True) -> None:
    if (not isinstance(imgs, np.ndarray) or imgs.ndim != 4 or imgs.shape[3] not in (3, 4)):
        raise ValueError("imgs must be a numpy.ndarray of shape (N, H, W, C), with C equal to 3 or 4.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_frames, H, W, C = imgs.shape
    if C == 3:
        pixel_format = "rgb24" if is_rgb else "bgr24"
    else:
        pixel_format = "rgba"
    ffmpeg_bin = resolve_ffmpeg_exe()
    ffmpeg = subprocess.Popen(
        [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            pixel_format,
            "-video_size",
            f"{W}x{H}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            f"{out_path}",
        ],
        stdin=subprocess.PIPE,
    )
    ffmpeg.stdin.write(imgs.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise IOError(f"Cannot open ffmpeg. Please check the output path and ensure ffmpeg is supported.")

    print(
        f"🎬 Video is saved to `{out_path}`, containing \033[94m{n_frames}\033[0m frames at {W}×{H} resolution and {fps} FPS."
    )
