"""Fork-only advanced features package (no license gates)."""

from anylabeling.views.labeling.advanced.bootstrap import (
    install_canvas_hooks,
    register_advanced_menu,
)

__all__ = ["install_canvas_hooks", "register_advanced_menu"]
