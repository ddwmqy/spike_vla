"""Evaluation entry point pinned to the code shipped in this experiment package."""

from __future__ import annotations

from pathlib import Path
import os
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ["REPO_DIR"]).resolve()
PACKAGE_CODE = PACKAGE_ROOT / "code"
PACKAGE_ADAPTER = PACKAGE_ROOT / "third_party" / "vla_adapter"

for source_root in (REPO_ROOT, PACKAGE_ADAPTER, PACKAGE_CODE):
    source_text = str(source_root)
    while source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)

# torch>=2.6 flipped torch.load's default to weights_only=True; LIBERO's
# benchmark loads its init-state pickles with the default and dies under the
# strict unpickler (numpy.core.multiarray._reconstruct). Restore the pre-2.6
# default for this eval process instead of dirtying the LIBERO checkout (the
# clean-tree gate pins it). All pickles consumed here are local artifacts we
# produced or the pinned LIBERO checkout ships, and this file is covered by
# the run fingerprint, so anchor and v2 share the exact same shim.
import torch  # noqa: E402

_original_torch_load = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load

from vla_adapter.rollout import main  # noqa: E402


if __name__ == "__main__":
    main()
