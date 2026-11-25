"""Trusted Demucs subprocess entry point with deterministic shift progress."""

from __future__ import annotations

import random
import sys
from types import SimpleNamespace


def main(arguments: list[str] | None = None) -> int:
    import demucs.apply as apply_module
    import demucs.separate as separate_module
    import torch
    from demucs.separate import main as demucs_main

    original_load = torch.load
    original_apply = separate_module.apply_model
    original_tqdm = apply_module.tqdm
    random.seed(0)
    torch.manual_seed(0)

    def apply_with_progress(model, mix, **kwargs):
        passes = len(getattr(model, "models", [model])) * max(
            1,
            int(kwargs.get("shifts", 1)),
        )
        completed = 0

        def progress_iter(items, **_options):
            nonlocal completed
            count = len(items)
            print(
                f"AUDIO_TOOLKIT_PROGRESS {int(100 * completed / passes)}",
                flush=True,
            )
            for index, item in enumerate(items):
                yield item
                value = int(
                    100
                    * (completed + (index + 1) / max(count, 1))
                    / passes
                )
                print(
                    f"AUDIO_TOOLKIT_PROGRESS {min(100, value)}",
                    flush=True,
                )
            completed += 1

        apply_module.tqdm = SimpleNamespace(tqdm=progress_iter)
        try:
            return original_apply(model, mix, **kwargs)
        finally:
            apply_module.tqdm = original_tqdm

    def load_verified_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_verified_checkpoint
    separate_module.apply_model = apply_with_progress
    try:
        demucs_main(list(arguments) if arguments is not None else sys.argv[1:])
    finally:
        torch.load = original_load
        separate_module.apply_model = original_apply
        apply_module.tqdm = original_tqdm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
