from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from PIL import Image, ImageTk

RESAMPLE = Image.Resampling.BILINEAR
log = logging.getLogger(__name__)


class ThreadBridge:

    EVENT_POLL_MS = 50

    def _init_bridge(self) -> None:
        self._events: queue.Queue[Callable[[], None]] = queue.Queue()
        self._drain_events()

    def post(self, fn: Callable, *args) -> None:
        self._events.put(lambda: fn(*args))

    def run_async(
        self,
        work: Callable[[], object],
        done: Callable[[object, Exception | None], None],
    ) -> None:

        def target():
            try:
                result, error = work(), None
            except Exception as exc:
                log.exception("Background task failed")
                result, error = None, exc
            self.post(done, result, error)

        threading.Thread(target=target, daemon=True).start()

    def _drain_events(self) -> None:
        try:
            while True:
                self._events.get_nowait()()
        except queue.Empty:
            pass
        except Exception:
            log.exception("UI event failed")
        self.after(self.EVENT_POLL_MS, self._drain_events)


class ImageCanvas(tk.Canvas):

    def __init__(
        self,
        master,
        *,
        allow_roi: bool = False,
        on_click: Callable[[float, float], None] | None = None,
        on_render: Callable[[ImageCanvas], None] | None = None,
        **kw,
    ):
        super().__init__(master, highlightthickness=0, **kw)
        self.allow_roi = allow_roi
        self.on_click = on_click
        self.on_render = on_render
        self.roi: tuple[int, int, int, int] | None = None
        self._image: Image.Image | None = None
        self._photo = None
        self._offset = (0, 0)
        self._scale = (1.0, 1.0)  # image pixels per canvas pixel
        self._size = (0, 0)  # displayed size in canvas pixels
        self._drag_start: tuple[int, int] | None = None
        self._drag_item = None
        self.bind("<Configure>", lambda _e: self._render())
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    @property
    def image(self) -> Image.Image | None:
        return self._image

    def show_image(self, image: Image.Image) -> None:
        self._image = image
        self._render()

    def clear(self) -> None:
        self._image = None
        self._photo = None
        self.delete("all")

    def clear_roi(self) -> None:
        self.roi = None
        self._render()

    def _render(self) -> None:
        self.delete("all")
        self._drag_item = None
        if self._image is None:
            return
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw < 2 or ch < 2:
            self.after(50, self._render)

            return
        iw, ih = self._image.size
        scale = min(cw / iw, ch / ih, 1.0)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        self._offset = ((cw - nw) // 2, (ch - nh) // 2)
        self._size = (nw, nh)
        self._scale = (iw / nw, ih / nh)
        self._photo = ImageTk.PhotoImage(self._image.resize((nw, nh), RESAMPLE))
        self.create_image(*self._offset, image=self._photo, anchor="nw")
        if self.roi:
            x, y, w, h = self.roi
            p0 = self.image_to_canvas((x, y))
            p1 = self.image_to_canvas((x + w, y + h))
            self.create_rectangle(*p0, *p1, outline="lime", width=2)
        if self.on_render:
            self.on_render(self)

    def image_to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        ox, oy = self._offset

        return ox + point[0] / self._scale[0], oy + point[1] / self._scale[1]

    def canvas_to_image(
        self, x: float, y: float, clamp: bool = False
    ) -> tuple[float, float] | None:
        ox, oy = self._offset
        nw, nh = self._size
        if clamp:
            x, y = min(max(x, ox), ox + nw), min(max(y, oy), oy + nh)
        elif not (ox <= x <= ox + nw and oy <= y <= oy + nh):
            return None

        return (x - ox) * self._scale[0], (y - oy) * self._scale[1]

    def _on_press(self, event) -> None:
        if self._image is None or self.canvas_to_image(event.x, event.y) is None:
            return
        if self.allow_roi:
            self._drag_start = (event.x, event.y)
            self._drag_item = self.create_rectangle(
                event.x, event.y, event.x, event.y, outline="yellow", width=2
            )
        elif self.on_click:
            self.on_click(*self.canvas_to_image(event.x, event.y))

    def _on_drag(self, event) -> None:
        if self._drag_start and self._drag_item:
            self.coords(self._drag_item, *self._drag_start, event.x, event.y)

    def _on_release(self, event) -> None:
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        a = self.canvas_to_image(x0, y0, clamp=True)
        b = self.canvas_to_image(event.x, event.y, clamp=True)
        w, h = abs(b[0] - a[0]), abs(b[1] - a[1])
        self.roi = (
            None
            if w < 4 or h < 4
            else (int(min(a[0], b[0])), int(min(a[1], b[1])), int(w), int(h))
        )
        self._render()


class ScrollableFrame(ttk.Frame):

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.body = ttk.Frame(self._canvas)
        self.body.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")
        self._canvas.bind(
            "<Configure>", lambda e: self._canvas.itemconfig(window, width=e.width)
        )
        self._canvas.configure(yscrollcommand=scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.bind(
            "<Enter>", lambda _e: self._canvas.bind_all("<MouseWheel>", self._on_wheel)
        )
        self.bind("<Leave>", lambda _e: self._canvas.unbind_all("<MouseWheel>"))

    def _on_wheel(self, event) -> None:
        if event.delta:
            self._canvas.yview_scroll(int(-event.delta / 120), "units")
