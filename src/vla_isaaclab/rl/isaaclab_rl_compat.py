"""Import Isaac Lab's RSL-RL wrappers without the other RL frameworks.

``isaaclab_rl/__init__.py`` imports its rl_games, sb3 and skrl wrappers, which
need those libraries (rl_games also needs the legacy ``gym``). Only RSL-RL is
installed here, so register ``isaaclab_rl`` as a bare package that only knows
its search path; ``isaaclab_rl.rsl_rl`` then imports on its own. Import this
module before anything from ``isaaclab_rl``.
"""

import importlib.util
import sys
import types


def _install() -> None:
    module = sys.modules.get("isaaclab_rl")
    if module is not None and hasattr(module, "rsl_rl"):
        return
    spec = importlib.util.find_spec("isaaclab_rl")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("isaaclab_rl is not installed")
    package = types.ModuleType("isaaclab_rl")
    package.__path__ = list(spec.submodule_search_locations)
    package.__spec__ = spec
    sys.modules["isaaclab_rl"] = package


_install()
