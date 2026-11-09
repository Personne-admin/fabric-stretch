from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from .widgets import ScrollableFrame


class RelaxPointsDialog:
    COLUMNS = 5

    def __init__(self, app, total: int, selected: set[int], on_ok: Callable[[set[int]], None]):
        dlg = tk.Toplevel(app)
        self.dlg = dlg
        dlg.title("Select Relaxation Points")
        dlg.geometry("400x350")
        dlg.transient(app)
        dlg.grab_set()

        scroll = ScrollableFrame(dlg)
        scroll.pack(fill="both", expand=True, padx=4, pady=4)
        self.vars = {}
        for i in range(total):
            self.vars[i] = tk.BooleanVar(value=i in selected)
            ttk.Checkbutton(scroll.body, text=f"#{i}", variable=self.vars[i]).grid(
                row=i // self.COLUMNS,
                column=i % self.COLUMNS,
                padx=5,
                pady=5,
                sticky="w",
            )

        bottom = ttk.Frame(dlg, padding=4)
        bottom.pack(fill="x", side="bottom")
        ttk.Button(bottom, text="All", width=6, command=lambda: self._set_all(True)).pack(side="left", padx=2)
        ttk.Button(bottom, text="None", width=6, command=lambda: self._set_all(False)).pack(
            side="left", padx=2
        )
        ttk.Button(bottom, text="OK", command=lambda: self._ok(on_ok)).pack(side="right", padx=4)

    def _set_all(self, value: bool):
        for var in self.vars.values():
            var.set(value)

    def _ok(self, on_ok: Callable[[set[int]], None]):
        on_ok({i for i, var in self.vars.items() if var.get()})
        self.dlg.destroy()
