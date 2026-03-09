"""Logging backends for Atlas.

Provides TensorBoard, Weights & Biases, and no-op loggers with a unified
interface. All loggers support scalars, histograms, text, images, and
hparams.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional


class NoopLogger:
    """No-op logger — silently discards all log calls."""

    def log(self, **kwargs: Any) -> None:
        pass

    def hist(self, tag: str, values: Any, step: int | None = None, **kw) -> None:
        pass

    def text(self, tag: str, text: str, step: int | None = None) -> None:
        pass

    def image(self, tag: str, img: Any, step: int | None = None) -> None:
        pass

    def graph(self, model: Any, example_input: Any) -> None:
        pass

    def hparams(self, hp: dict, metrics: dict | None = None) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class TBLogger(NoopLogger):
    """TensorBoard logger.

    Auto-routes values: scalar tensors → ``add_scalar``, multi-element
    tensors → ``add_histogram``, strings → ``add_text``.

    Args:
        out_dir: Root output directory (a timestamped sub-dir is created).
        flush_secs: TensorBoard flush interval.
        run_name: Optional run name override.
    """

    def __init__(
        self,
        out_dir: str,
        flush_secs: int = 10,
        run_name: str | None = None,
    ) -> None:
        self.w = None
        self._hparams_logged = False
        run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(out_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.w = SummaryWriter(log_dir=str(run_dir), flush_secs=flush_secs)
        except Exception as e:
            print(f"[TBLogger] TensorBoard unavailable: {e}. Logging disabled.")

    def log(self, step: int | None = None, **kv: Any) -> None:
        if not self.w:
            return
        for k, v in kv.items():
            try:
                import torch
                import numpy as np
                is_torch = isinstance(v, torch.Tensor)
                is_np = isinstance(v, np.ndarray)
                if is_torch or is_np:
                    numel = int(v.numel() if is_torch else v.size)
                    if numel == 1:
                        self.w.add_scalar(k, float(v.item() if is_torch else v), global_step=step)
                    elif numel <= 2048:
                        self.w.add_histogram(k, v.detach().cpu() if is_torch else v, global_step=step)
                    continue
            except Exception:
                pass
            try:
                self.w.add_scalar(k, float(v), global_step=step)
            except Exception:
                pass

    def hist(self, tag: str, values: Any, step: int | None = None, **kw) -> None:
        if not self.w:
            return
        try:
            import torch
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu()
            self.w.add_histogram(tag, values, global_step=step)
        except Exception:
            pass

    def text(self, tag: str, text: str, step: int | None = None) -> None:
        if self.w:
            try:
                self.w.add_text(tag, text, global_step=step)
            except Exception:
                pass

    def image(self, tag: str, img: Any, step: int | None = None) -> None:
        if self.w:
            try:
                fmt = "CHW" if getattr(img, "ndim", 0) == 3 and img.shape[0] in (1, 3) else "HWC"
                self.w.add_image(tag, img, global_step=step, dataformats=fmt)
            except Exception:
                pass

    def graph(self, model: Any, example_input: Any) -> None:
        if not self.w:
            return
        try:
            if not isinstance(example_input, tuple):
                example_input = (example_input,)
            self.w.add_graph(model, example_input)
        except Exception:
            pass

    def hparams(self, hp: dict, metrics: dict | None = None) -> None:
        if not self.w or self._hparams_logged:
            return
        try:
            self.w.add_hparams(hp, metrics or {}, run_name="_hparams")
            self._hparams_logged = True
        except Exception:
            pass

    def flush(self) -> None:
        if self.w:
            try:
                self.w.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self.w:
            try:
                self.w.close()
            except Exception:
                pass


class WBLogger(NoopLogger):
    """Weights & Biases logger.

    Args:
        project: W&B project name.
        run_name: Optional run name.
    """

    def __init__(self, project: str = "atlas", run_name: str | None = None) -> None:
        self.wb = None
        try:
            import wandb
            wandb.init(project=project, name=run_name)
            self.wb = wandb
        except Exception:
            pass

    def log(self, **kv: Any) -> None:
        if self.wb:
            self.wb.log(kv)

    def close(self) -> None:
        if self.wb:
            try:
                self.wb.finish()
            except Exception:
                pass


def init_logger(backend: str = "tensorboard", out_dir: str = "runs") -> NoopLogger:
    """Initialize a logger by backend name.

    Args:
        backend: ``"tensorboard"``, ``"wandb"``, or ``"none"``.
        out_dir: Output directory (TensorBoard only).

    Returns:
        A logger instance.
    """
    if backend == "tensorboard":
        tb = TBLogger(out_dir)
        return tb if tb.w is not None else NoopLogger()
    elif backend == "wandb":
        return WBLogger()
    return NoopLogger()
