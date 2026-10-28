"""
seed.py -- single seeding entry point (ROADMAP, Phase 1 -- seeding).

Every entrypoint must call `set_seed(cfg.seed)` before touching data or model
code. The reference notebook seeds nothing: it draws destination IPs with
`random.randint`, samples subgraphs with `random.sample`, and samples node
batches with `torch.randint`. Each of those is a separate RNG stream, so all
three must be pinned for two runs to agree.

`PYTHONHASHSEED` is deliberately NOT set here -- it only takes effect at
interpreter start. The launcher sets it:

    PYTHONHASHSEED=0 python main.py --seed 42

Without it, `list(set(...))` in data.py yields a different column order per
process, and any checkpoint becomes unusable across runs (see the roadmap's
state section and its evidence appendix).
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline uses, and ask torch for determinism.

    Args:
        seed: the run seed. Record it in the run log; a run is only
            reproducible together with its PYTHONHASHSEED and git SHA.
        deterministic: set torch's deterministic algorithm mode. Kept on by
            default; `warn_only=True` downgrades unsupported ops (some DGL
            scatter kernels) from errors to warnings, which is what we want --
            we would rather have a warning than a crashed training run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def describe() -> dict:
    """Return the seeding-relevant environment, for the run manifest."""
    return {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def warn_if_unpinned() -> None:
    """Warn when PYTHONHASHSEED is unset -- the run will not be reproducible."""
    if os.environ.get("PYTHONHASHSEED") is None:
        import warnings

        warnings.warn(
            "PYTHONHASHSEED is unset. Feature-column order is derived from a "
            "set() and will differ between processes, so this run cannot be "
            "compared with, or reused by, any other run. Relaunch as: "
            "PYTHONHASHSEED=0 python <entrypoint> ...",
            RuntimeWarning,
            stacklevel=2,
        )
