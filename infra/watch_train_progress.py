#!/usr/bin/env python3
"""Watch AHA-WAM train logs and refresh live progress charts on dataset disk.

Parses:
  [train] epoch=0 step=10/2500 loss=1.2345 loss_action_prior=... lr=... grad_norm=... eta=...

Artifacts under --out-dir:
  metrics.jsonl              append-only history
  status.txt                 one-line latest status
  progress_live.svg/png      loss vs step (simple)
  dashboard_live.svg/png     multi-panel: loss / action loss / lr / grad_norm / speed
  snapshots/step_*.{svg,png} periodic snapshots (if --snapshot-every > 0)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TRAIN_RE = re.compile(
    r"(?:\[train\]\s+)?epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)/(?P<max_steps>\d+)"
    r"(?:\s+loss=(?P<loss>[-+eE0-9.]+))?"
)
DETAIL_RE = re.compile(r"(?P<key>loss_\w+|grad_norm|lr)=(?P<val>[-+eE0-9.]+)")
ETA_RE = re.compile(r"eta=(?P<eta>\S+)")
SPEED_RE = re.compile(r"speed=(?P<speed>[-+eE0-9.]+)\s+step/s")
# Rich logger wraps the trainer line across several physical lines.
LOSS_ONLY_RE = re.compile(r"\bloss=(?P<loss>[-+eE0-9.]+)\b")


def parse_train_line(line: str) -> Optional[Dict]:
    """Parse a single physical line; may be incomplete if rich wrapped the record."""
    m = TRAIN_RE.search(line)
    if not m:
        return None
    row = {
        "ts": time.time(),
        "epoch": int(m.group("epoch")),
        "step": int(m.group("step")),
        "max_steps": int(m.group("max_steps")),
    }
    if m.groupdict().get("loss"):
        row["loss"] = float(m.group("loss"))
    for dm in DETAIL_RE.finditer(line):
        try:
            row[dm.group("key")] = float(dm.group("val"))
        except ValueError:
            pass
    em = ETA_RE.search(line)
    if em:
        row["eta"] = em.group("eta")
    sm = SPEED_RE.search(line)
    if sm:
        try:
            row["steps_per_sec"] = float(sm.group("speed"))
        except ValueError:
            pass
    return row


def parse_train_buffer(buf: str) -> List[Dict]:
    """Parse rich-wrapped train logs: join lines until a full step record appears."""
    # Split on step headers, keep following continuation lines with that step.
    rows: List[Dict] = []
    # Find each "epoch=X step=Y/Z" and take a window of following text until next epoch= or ~800 chars
    for m in TRAIN_RE.finditer(buf):
        start = m.start()
        # window: from this match to next epoch= or end
        nxt = TRAIN_RE.search(buf, m.end())
        end = nxt.start() if nxt else min(len(buf), m.end() + 900)
        chunk = buf[start:end]
        row = {
            "ts": time.time(),
            "epoch": int(m.group("epoch")),
            "step": int(m.group("step")),
            "max_steps": int(m.group("max_steps")),
        }
        lm = LOSS_ONLY_RE.search(chunk)
        if lm:
            row["loss"] = float(lm.group("loss"))
        else:
            # incomplete record (header only) — skip until loss appears
            continue
        for dm in DETAIL_RE.finditer(chunk):
            try:
                row[dm.group("key")] = float(dm.group("val"))
            except ValueError:
                pass
        # prefer loss_action* as display if total loss missing (shouldn't)
        if "loss" not in row:
            for k in ("loss_action", "loss_action_prior"):
                if k in row:
                    row["loss"] = row[k]
                    break
        em = ETA_RE.search(chunk)
        if em:
            row["eta"] = em.group("eta")
        sm = SPEED_RE.search(chunk)
        if sm:
            try:
                row["steps_per_sec"] = float(sm.group("speed"))
            except ValueError:
                pass
        rows.append(row)
    return rows


def _series(points: Sequence[Dict], key: str) -> List[float]:
    out = []
    for p in points:
        if key in p and p[key] is not None:
            try:
                out.append(float(p[key]))
            except (TypeError, ValueError):
                pass
    return out


def render_svg(points: List[Dict], out_svg: Path, title: str) -> None:
    if not points:
        return
    w, h = 960, 420
    pad_l, pad_r, pad_t, pad_b = 70, 30, 70, 50
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    steps = [p["step"] for p in points]
    losses = [p["loss"] for p in points]
    max_steps = max(int(points[-1].get("max_steps") or 0), max(steps), 1)
    ymin = min(losses)
    ymax = max(losses)
    if abs(ymax - ymin) < 1e-8:
        ymax = ymin + 1.0
    pad_y = 0.05 * (ymax - ymin)
    ymin -= pad_y
    ymax += pad_y

    def x_of(s: float) -> float:
        return pad_l + plot_w * (s / max_steps)

    def y_of(v: float) -> float:
        return pad_t + plot_h * (1.0 - (v - ymin) / (ymax - ymin))

    poly = " ".join(f"{x_of(s):.1f},{y_of(v):.1f}" for s, v in zip(steps, losses))
    last = points[-1]
    eta = last.get("eta", "?")
    speed = last.get("steps_per_sec")
    speed_s = f"{speed:.2f} step/s" if isinstance(speed, float) else "?"
    pct = 100.0 * last["step"] / max_steps
    header = (
        f"{title} | step {last['step']}/{max_steps} ({pct:.1f}%) | "
        f"loss={last['loss']:.4f} | {speed_s} | eta={eta}"
    )

    ticks = []
    for i in range(5):
        v = ymin + (ymax - ymin) * i / 4.0
        yy = y_of(v)
        ticks.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w-pad_r}" y2="{yy:.1f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
            f'<text x="{pad_l-8}" y="{yy+4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#444">{v:.3f}</text>'
        )
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{pad_l}" y="28" font-size="16" font-family="DejaVu Sans, Arial, sans-serif" fill="#1f4e79">{header}</text>
  <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#cccccc"/>
  {''.join(ticks)}
  <polyline fill="none" stroke="#1f4e79" stroke-width="2.5" points="{poly}"/>
  <line x1="{x_of(max_steps):.1f}" y1="{pad_t}" x2="{x_of(max_steps):.1f}" y2="{pad_t+plot_h}" stroke="#999" stroke-dasharray="4 4"/>
  <text x="{pad_l + plot_w/2:.1f}" y="{h-12}" text-anchor="middle" font-size="13" fill="#333">optimizer step</text>
  <text x="18" y="{pad_t + plot_h/2:.1f}" transform="rotate(-90 18,{pad_t + plot_h/2:.1f})" text-anchor="middle" font-size="13" fill="#333">loss</text>
  <text x="{pad_l}" y="{h-28}" font-size="11" fill="#666">updated {time.strftime('%Y-%m-%d %H:%M:%S')} | points={len(points)}</text>
</svg>
"""
    tmp = out_svg.parent / (out_svg.stem + ".__tmp__.svg")
    tmp.write_text(svg, encoding="utf-8")
    os.replace(tmp, out_svg)


def render_png(points: List[Dict], out_png: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    steps = [p["step"] for p in points]
    losses = [p["loss"] for p in points]
    max_steps = int(points[-1].get("max_steps") or 0)
    last = points[-1]
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    ax.plot(steps, losses, color="#1f4e79", linewidth=1.8, marker="o", markersize=2.5)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title(
        f"{title}\nstep {last['step']}/{max_steps} loss={last['loss']:.4f} eta={last.get('eta', '?')}"
    )
    if max_steps > 0:
        ax.set_xlim(0, max(max_steps, max(steps)))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    tmp = out_png.parent / (out_png.stem + ".__tmp__.png")
    fig.savefig(tmp)
    plt.close(fig)
    os.replace(tmp, out_png)


def render_dashboard_png(points: List[Dict], out_png: Path, title: str) -> None:
    """Multi-panel dashboard: loss, action loss, lr, grad_norm, speed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not points:
        return
    steps = [p["step"] for p in points]
    max_steps = max(int(points[-1].get("max_steps") or 0), max(steps), 1)
    last = points[-1]
    pct = 100.0 * last["step"] / max_steps

    panels = [
        ("loss", "loss", "#1f4e79"),
        ("loss_action_prior", "loss_action_prior", "#2a9d8f"),
        ("lr", "lr", "#e76f51"),
        ("grad_norm", "grad_norm", "#9b5de5"),
        ("steps_per_sec", "speed (step/s)", "#457b9d"),
    ]
    # Drop empty panels except total loss
    active = []
    for key, label, color in panels:
        if key == "loss" or any(key in p for p in points):
            active.append((key, label, color))

    n = len(active)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n), dpi=120, sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (key, label, color) in zip(axes, active):
        ys = []
        xs = []
        for p in points:
            if key in p:
                try:
                    ys.append(float(p[key]))
                    xs.append(int(p["step"]))
                except (TypeError, ValueError):
                    pass
        if not xs:
            ax.text(0.5, 0.5, f"no {label} yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(label)
            continue
        ax.plot(xs, ys, color=color, linewidth=1.6)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_steps)
    axes[-1].set_xlabel("optimizer step")
    fig.suptitle(
        f"{title} | step {last['step']}/{max_steps} ({pct:.1f}%) | "
        f"loss={last['loss']:.4f} | eta={last.get('eta', '?')} | "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    tmp = out_png.parent / (out_png.stem + ".__tmp__.png")
    fig.savefig(tmp)
    plt.close(fig)
    os.replace(tmp, out_png)


def render_dashboard_svg(points: List[Dict], out_svg: Path, title: str) -> None:
    """Lightweight multi-panel SVG (no matplotlib required)."""
    if not points:
        return
    panels = [
        ("loss", "loss", "#1f4e79"),
        ("loss_action_prior", "action", "#2a9d8f"),
        ("lr", "lr", "#e76f51"),
        ("grad_norm", "grad", "#9b5de5"),
        ("steps_per_sec", "speed", "#457b9d"),
    ]
    active = []
    for key, label, color in panels:
        vals = _series(points, key)
        if key == "loss" or vals:
            active.append((key, label, color, vals if vals else _series(points, "loss")))

    panel_h = 110
    pad_top = 56
    w = 1000
    h = pad_top + panel_h * len(active) + 36
    pad_l, pad_r = 70, 24
    plot_w = w - pad_l - pad_r
    steps = [p["step"] for p in points]
    max_steps = max(int(points[-1].get("max_steps") or 0), max(steps), 1)
    last = points[-1]
    pct = 100.0 * last["step"] / max_steps
    header = (
        f"{title} | step {last['step']}/{max_steps} ({pct:.1f}%) | "
        f"loss={last['loss']:.4f} | eta={last.get('eta', '?')}"
    )

    parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad_l}" y="28" font-size="15" font-family="DejaVu Sans, Arial, sans-serif" fill="#1f4e79">{header}</text>',
        f'<text x="{pad_l}" y="48" font-size="11" fill="#666">updated {time.strftime("%Y-%m-%d %H:%M:%S")} | points={len(points)}</text>',
    ]

    for i, (key, label, color, _) in enumerate(active):
        top = pad_top + i * panel_h
        plot_top = top + 18
        plot_h = panel_h - 30
        xs, ys = [], []
        for p in points:
            if key not in p and key != "loss":
                continue
            try:
                ys.append(float(p.get(key, p["loss"])))
                xs.append(float(p["step"]))
            except (TypeError, ValueError, KeyError):
                continue
        if not ys:
            continue
        ymin, ymax = min(ys), max(ys)
        if abs(ymax - ymin) < 1e-12:
            ymax = ymin + 1.0
        pad_y = 0.08 * (ymax - ymin)
        ymin -= pad_y
        ymax += pad_y

        def x_of(s: float) -> float:
            return pad_l + plot_w * (s / max_steps)

        def y_of(v: float) -> float:
            return plot_top + plot_h * (1.0 - (v - ymin) / (ymax - ymin))

        poly = " ".join(f"{x_of(s):.1f},{y_of(v):.1f}" for s, v in zip(xs, ys))
        parts.append(
            f'<text x="{pad_l}" y="{top+12}" font-size="12" fill="#333">{label} '
            f'(last={ys[-1]:.4g})</text>'
        )
        parts.append(
            f'<rect x="{pad_l}" y="{plot_top}" width="{plot_w}" height="{plot_h}" '
            f'fill="#fafafa" stroke="#ddd"/>'
        )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{poly}"/>'
        )

    parts.append("</svg>")
    tmp = out_svg.parent / (out_svg.stem + ".__tmp__.svg")
    tmp.write_text("\n".join(parts), encoding="utf-8")
    os.replace(tmp, out_svg)


def follow_files(
    paths: List[Path],
    stop_flag: Path,
    interval: float,
    out_dir: Path,
    title: str,
    *,
    snapshot_every: int = 0,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_dir = out_dir / "snapshots"
    if snapshot_every > 0:
        snap_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    svg_path = out_dir / "progress_live.svg"
    png_path = out_dir / "progress_live.png"
    dash_svg = out_dir / "dashboard_live.svg"
    dash_png = out_dir / "dashboard_live.png"
    status_path = out_dir / "status.txt"
    readme = out_dir / "README.txt"
    readme.write_text(
        "Live train progress (auto-refreshed)\n"
        "- status.txt              latest one-liner (tail -f this)\n"
        "- progress_live.svg/png   loss curve\n"
        "- dashboard_live.svg/png  multi-panel loss / action / lr / grad / speed\n"
        "- snapshots/step_*.png    periodic snapshots\n"
        "- metrics.jsonl           full history\n",
        encoding="utf-8",
    )

    seen_steps = set()
    snapped_steps = set()
    points: List[Dict] = []
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in row and "loss" in row:
                seen_steps.add(int(row["step"]))
                points.append(row)

    offsets = {str(p): 0 for p in paths}
    last_render = 0.0
    # Keep a small tail buffer so rich-wrapped multi-line records can be parsed.
    tail_buf = {str(p): "" for p in paths}
    status_path.write_text(
        f"{time.strftime('%F %T')} watcher started; waiting for [train] logs...\n",
        encoding="utf-8",
    )

    def render_all() -> None:
        render_svg(points, svg_path, title)
        render_png(points, png_path, title)
        render_dashboard_svg(points, dash_svg, title)
        render_dashboard_png(points, dash_png, title)

    def maybe_snapshot(step: int) -> None:
        if snapshot_every <= 0 or not points:
            return
        if step % snapshot_every != 0:
            return
        if step in snapped_steps:
            return
        snapped_steps.add(step)
        render_svg(points, snap_dir / f"step_{step:06d}.svg", title)
        render_png(points, snap_dir / f"step_{step:06d}.png", title)
        render_dashboard_png(points, snap_dir / f"dashboard_{step:06d}.png", title)

    def ingest_rows(new_rows: List[Dict]) -> None:
        for row in new_rows:
            step = int(row["step"])
            if step in seen_steps:
                continue
            if "loss" not in row:
                continue
            seen_steps.add(step)
            points.append(row)
            metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics_f.flush()
            max_steps = max(int(row.get("max_steps") or 1), 1)
            pct = 100.0 * step / max_steps
            status_path.write_text(
                f"{time.strftime('%F %T')} step={step}/{row.get('max_steps')} "
                f"({pct:.1f}%) loss={row['loss']:.4f} "
                f"lr={row.get('lr', '?')} eta={row.get('eta', '?')}\n",
                encoding="utf-8",
            )
            maybe_snapshot(step)

    with metrics_path.open("a", encoding="utf-8") as metrics_f:
        while True:
            if stop_flag.is_file():
                if points:
                    render_all()
                    maybe_snapshot(int(points[-1]["step"]))
                status_path.write_text(
                    f"STOPPED {time.strftime('%F %T')} points={len(points)}\n",
                    encoding="utf-8",
                )
                break
            for p in paths:
                key = str(p)
                if not p.is_file():
                    continue
                size = p.stat().st_size
                if size < offsets[key]:
                    offsets[key] = 0
                    tail_buf[key] = ""
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    f.seek(offsets[key])
                    chunk = f.read()
                    offsets[key] = f.tell()
                if not chunk:
                    continue
                # Keep last ~8k so a wrapped record spanning reads still parses.
                combined = (tail_buf[key] + chunk)[-12000:]
                tail_buf[key] = combined[-4000:]
                ingest_rows(parse_train_buffer(combined))
            now = time.time()
            if points and now - last_render >= interval:
                render_all()
                last_render = now
            time.sleep(min(interval, 5.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--title", default="train progress")
    ap.add_argument("--stop-flag", default="")
    ap.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="If >0, also write snapshots/step_XXXXXX.{svg,png} every N steps",
    )
    args = ap.parse_args()
    logs = [Path(x) for x in args.log]
    if not logs:
        raise SystemExit("need at least one --log")
    out_dir = Path(args.out_dir)
    stop_flag = Path(args.stop_flag) if args.stop_flag else (out_dir / "WATCHER_STOP")
    follow_files(
        logs,
        stop_flag,
        float(args.interval),
        out_dir,
        args.title,
        snapshot_every=int(args.snapshot_every),
    )


if __name__ == "__main__":
    main()
