"""Manual check for overlay.py (§21.7). Must run on main thread (Tk).
Shows the blurred overlay for 3 seconds, then hides it. Run:
python tests/manual_check_overlay.py
Expect: your screen appears fullscreen, heavily blurred, with a banner
at the top, then returns to normal after 3 seconds."""
from __future__ import annotations

import tkinter as tk

from ssa.overlay.overlay import Overlay
from ssa.platform_utils import set_dpi_aware


def main() -> None:
    set_dpi_aware()
    root = tk.Tk()
    root.withdraw()
    overlay = Overlay(root, blur_kernel=99, downscale=0.25)

    def show_then_hide() -> None:
        overlay.show("Privacy alert: someone is looking at your screen (manual test)")
        root.after(3000, lambda: (overlay.hide(), root.quit()))

    root.after(500, show_then_hide)
    root.mainloop()


if __name__ == "__main__":
    main()

