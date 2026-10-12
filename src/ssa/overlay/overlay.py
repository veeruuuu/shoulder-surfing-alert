"""Tk blur overlay (§5.4). Must run on the main thread only (§5.5, §21.6)."""
from __future__ import annotations

import tkinter as tk
from typing import Optional

import cv2
import mss
import numpy as np
from PIL import Image, ImageTk


class Overlay:
    """Fullscreen blurred overlay shown on ALERT_ON/AWAY_ON, hidden on
    ALERT_OFF/AWAY_OFF. Created and used only on the main thread."""

    def __init__(self, root: tk.Tk, blur_kernel: int = 99, downscale: float = 0.25) -> None:
        self._root = root
        self._blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self._downscale = downscale
        self._window: Optional[tk.Toplevel] = None
        self._photo: Optional[ImageTk.PhotoImage] = None  # keep a reference, Tk needs it
        self._banner_label: Optional[tk.Label] = None

    def show(self, banner_text: str) -> None:
        if self._window is not None:
            self._update_banner(banner_text)
            return

        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()

        blurred = self._capture_and_blur(screen_w, screen_h)

        win = tk.Toplevel(self._root)
        win.geometry(f"{screen_w}x{screen_h}+0+0")
        win.attributes("-topmost", True)
        win.overrideredirect(True)
        win.configure(bg="black")
        win.bind("<Escape>", lambda e: self.hide())

        img = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(img)

        canvas = tk.Canvas(
            win, width=screen_w, height=screen_h, highlightthickness=0, bg="black"
        )
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, image=self._photo, anchor="nw")

        self._banner_label = tk.Label(
            win, text=banner_text, font=("Segoe UI", 18, "bold"),
            fg="white", bg="#222222", padx=20, pady=10,
        )
        self._banner_label.place(relx=0.5, rely=0.08, anchor="center")

        win.focus_force()
        self._window = win

    def hide(self) -> None:
        if self._window is not None:
            self._window.destroy()
            self._window = None
            self._photo = None
            self._banner_label = None

    def is_visible(self) -> bool:
        return self._window is not None

    def _update_banner(self, banner_text: str) -> None:
        if self._banner_label is not None:
            self._banner_label.config(text=banner_text)

    def _capture_and_blur(self, target_w: int, target_h: int) -> np.ndarray:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            shot = sct.grab(monitor)
            frame = np.array(shot)[:, :, :3]  # BGRA -> BGR

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, int(w * self._downscale)), max(1, int(h * self._downscale))))
        blurred_small = cv2.GaussianBlur(small, (self._blur_kernel, self._blur_kernel), 0)
        blurred = cv2.resize(blurred_small, (target_w, target_h))
        return blurred
