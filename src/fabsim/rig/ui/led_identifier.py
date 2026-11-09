from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

ROLES = [
    ("unused", "Unused"),
    ("pol_a_0", "Pol A1"),
    ("pol_a_1", "Pol A2"),
    ("pol_a_2", "Pol A3"),
    ("pol_b_0", "Pol B1"),
    ("pol_b_1", "Pol B2"),
    ("pol_b_2", "Pol B3"),
    ("diffuse", "Diffuse"),
]
N_BITS = 8


def bit_role(cfg: dict, bit: int) -> str:
    for i, b in enumerate(cfg.get("pol_a_bits", [])):
        if b == bit:
            return f"pol_a_{i}"
    for i, b in enumerate(cfg.get("pol_b_bits", [])):
        if b == bit:
            return f"pol_b_{i}"
    if cfg.get("diffuse_mask", 0) & (1 << bit):
        return "diffuse"

    return "unused"


def roles_to_mapping(roles: dict[int, str]) -> tuple[list[int], list[int], int]:
    role_to_bit = {role: bit for bit, role in roles.items()}
    sets = {}
    for prefix, name in (("pol_a", "A"), ("pol_b", "B")):
        present = [f"{prefix}_{i}" in role_to_bit for i in range(3)]
        if any(present) and not all(present):
            missing = ", ".join(
                f"{name}{i + 1}" for i, ok in enumerate(present) if not ok
            )
            raise ValueError(
                f"Set {name} is incomplete (missing {missing}): assign all of {name}1-{name}3 or none."
            )
        sets[prefix] = (
            [role_to_bit[f"{prefix}_{i}"] for i in range(3)] if all(present) else []
        )
    diffuse = 0
    for bit, role in roles.items():
        if role == "diffuse":
            diffuse |= 1 << bit

    return sets["pol_a"], sets["pol_b"], diffuse


class LedIdentifierDialog:
    def __init__(self, app):
        self.app = app
        self.ctrl = app.ctrl
        self.test_mask = 0
        dlg = tk.Toplevel(app)
        self.dlg = dlg
        dlg.title("LED Matrix Setup")
        dlg.geometry("720x450")
        dlg.transient(app)
        dlg.grab_set()
        dlg.protocol("WM_DELETE_WINDOW", self._close)

        ttk.Label(
            dlg,
            text=(
                "Turn on each bit to identify the physical LED it corresponds to, "
                "then assign its role. Roles A1-A3 and B1-B3 are mutually exclusive, "
                "while Diffuse can be assigned to multiple bits."
            ),
            wraplength=680,
            font=("", 10),
        ).pack(padx=12, pady=(12, 6), anchor="w")

        p_row = ttk.Frame(dlg)
        p_row.pack(fill="x", padx=12, pady=6)
        ttk.Label(p_row, text="Test Power:").pack(side="left")
        self.power_var = tk.IntVar(value=128)
        ttk.Scale(
            p_row,
            from_=0,
            to=255,
            variable=self.power_var,
            orient="horizontal",
            command=self._on_power,
        ).pack(side="left", fill="x", expand=True, padx=8)

        grid = ttk.LabelFrame(dlg, text="Bit Assignments", padding=10)
        grid.pack(fill="both", expand=True, padx=12, pady=6)
        for col, text in enumerate(["Bit", "Test"] + [label for _, label in ROLES]):
            ttk.Label(grid, text=text, font=("", 9, "bold")).grid(
                row=0, column=col, padx=4, pady=4
            )

        self.bit_vars: dict[int, tk.StringVar] = {}
        for bit in range(N_BITS):
            self.bit_vars[bit] = tk.StringVar(value=bit_role(app.led_config, bit))
            ttk.Label(grid, text=f"Bit {bit}").grid(
                row=bit + 1, column=0, padx=4, pady=2
            )
            ttk.Button(
                grid, text="ON", width=6, command=lambda b=bit: self._flash(b)
            ).grid(row=bit + 1, column=1, padx=4, pady=2)
            for col, (value, _) in enumerate(ROLES):
                ttk.Radiobutton(
                    grid,
                    variable=self.bit_vars[bit],
                    value=value,
                    command=lambda b=bit: self._on_role_change(b),
                ).grid(row=bit + 1, column=col + 2, padx=2, pady=2)

        btns = ttk.Frame(dlg)
        btns.pack(fill="x", padx=12, pady=12)
        ttk.Button(btns, text="All ON", command=lambda: self._light(0xFF)).pack(
            side="left", padx=2
        )
        ttk.Button(btns, text="All OFF", command=self._all_off).pack(
            side="left", padx=2
        )
        ttk.Button(btns, text="Save & Close", command=self._save_and_close).pack(
            side="right", padx=2
        )
        ttk.Button(btns, text="Cancel", command=self._close).pack(side="right", padx=2)

    def _light(self, mask: int):
        self.test_mask = mask
        self.ctrl.set_power(self.power_var.get())
        self.ctrl.set_leds(mask)

    def _on_power(self, *_):
        if self.test_mask and self.ctrl.connected:
            self.ctrl.set_power(self.power_var.get())

    def _flash(self, bit: int):
        self._light(1 << bit)
        self.app.append_log(f"Identify: bit {bit} (mask 0x{1 << bit:02X})")

    def _all_off(self):
        self.test_mask = 0
        self.ctrl.lights_off()

    def _on_role_change(self, changed_bit: int):
        role = self.bit_vars[changed_bit].get()
        if role not in ("unused", "diffuse"):
            for bit, var in self.bit_vars.items():
                if bit != changed_bit and var.get() == role:
                    var.set("unused")

    def _save_and_close(self):
        try:
            pol_a, pol_b, diffuse = roles_to_mapping(
                {bit: var.get() for bit, var in self.bit_vars.items()}
            )
        except ValueError as exc:
            messagebox.showerror("Incomplete assignment", str(exc), parent=self.dlg)

            return
        self.app.update_led_mapping(pol_a, pol_b, diffuse)
        self._close()

    def _close(self):
        if self.ctrl.connected:
            self.ctrl.lights_off()
        self.dlg.destroy()
