"""
browse_dataset.py
-----------------
Run this script to visually browse images from any QuickDraw .npy file
and find the indices of samples you want to use in the AI Draws mode.

Usage:
    python browse_dataset.py

Controls:
    ← / →       Previous / Next page (20 images per page)
    Click image  Print its index to the console (copy into PICKED_SAMPLES)
    Category     Type a category name in the entry box and press Enter
    Q / Escape   Quit
"""

import os
import numpy as np
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk

DATA_PATH = "data"
CATEGORIES = [
    "circle", "square", "triangle", "tree", "house",
    "star", "cloud", "moon", "sun", "book",
    "beach", "clock", "fish", "airplane", "leaf",
    "calendar", "pencil", "guitar", "cup", "flower",
    "cat", "car", "umbrella", "bicycle", "ladder",
    "key", "dog", "lightning", "chair", "hat"
]

GRID_COLS   = 10
GRID_ROWS   = 5
PER_PAGE    = GRID_COLS * GRID_ROWS   # 50 images per page
THUMB_SIZE  = 80                       # pixels per thumbnail
PAD         = 6
BG          = "#07070f"
PANEL       = "#121220"
CARD        = "#1a1a2e"
BORDER      = "#2a2a4a"
ACCENT      = "#6c63ff"
ACCENT2     = "#ff6584"
TEXT_HI     = "#f0eeff"
TEXT_MID    = "#a09cc0"
TEXT_LO     = "#4a4870"
SUCCESS     = "#43e8a0"
WARNING     = "#ffd166"


class BrowseApp:
    def __init__(self, root):
        self.root     = root
        self.root.title("QuickDraw Dataset Browser")
        self.root.configure(bg=BG)
        self.root.state("zoomed")
        self.root.update_idletasks()
        self.W = self.root.winfo_screenwidth()
        self.H = self.root.winfo_screenheight()
        self.root.geometry(f"{self.W}x{self.H}")

        self.category  = CATEGORIES[0]
        self.raw_data  = None
        self.page      = 0
        self.picked    = {}   # {category: [index, ...]}
        self._thumbs   = []   # keep PhotoImage refs alive

        f_title  = tkfont.Font(family="Georgia",     size=18, weight="bold")
        f_label  = tkfont.Font(family="Courier New", size=10)
        f_small  = tkfont.Font(family="Courier New", size=8)
        f_btn    = tkfont.Font(family="Courier New", size=10, weight="bold")
        self.f_label = f_label
        self.f_small = f_small

        # ── Top bar ────────────────────────────────────────────────
        top = tk.Frame(root, bg=PANEL, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="QuickDraw Browser", font=f_title,
                 bg=PANEL, fg=ACCENT).pack(side="left", padx=20)

        # Category selector
        tk.Label(top, text="Category:", font=f_label,
                 bg=PANEL, fg=TEXT_MID).pack(side="left", padx=(20, 4))

        self.cat_var = tk.StringVar(value=self.category)
        cat_menu = tk.OptionMenu(top, self.cat_var, *CATEGORIES,
                                 command=self._on_category_change)
        cat_menu.config(bg=CARD, fg=TEXT_HI, font=f_label,
                        activebackground=BORDER, highlightthickness=0,
                        relief="flat")
        cat_menu["menu"].config(bg=CARD, fg=TEXT_HI, font=f_label)
        cat_menu.pack(side="left")

        # Page nav
        tk.Label(top, text="  Page:", font=f_label,
                 bg=PANEL, fg=TEXT_MID).pack(side="left", padx=(20, 4))

        self.page_var = tk.StringVar(value="1")
        page_entry = tk.Entry(top, textvariable=self.page_var,
                              font=f_label, bg=CARD, fg=TEXT_HI,
                              width=5, relief="flat",
                              insertbackground=ACCENT)
        page_entry.pack(side="left")
        page_entry.bind("<Return>", self._on_page_entry)

        self.total_pages_lbl = tk.Label(top, text="/ ?",
                                        font=f_label, bg=PANEL, fg=TEXT_LO)
        self.total_pages_lbl.pack(side="left", padx=4)

        prev_btn = tk.Label(top, text="  ◀ PREV  ", font=f_btn,
                            bg=CARD, fg=TEXT_MID, cursor="hand2",
                            padx=8, pady=4)
        prev_btn.pack(side="left", padx=8)
        prev_btn.bind("<Button-1>", lambda e: self._change_page(-1))

        next_btn = tk.Label(top, text="  NEXT ▶  ", font=f_btn,
                            bg=CARD, fg=TEXT_MID, cursor="hand2",
                            padx=8, pady=4)
        next_btn.pack(side="left")
        next_btn.bind("<Button-1>", lambda e: self._change_page(1))

        # Print picked button
        print_btn = tk.Label(top, text="  📋 PRINT PICKED  ", font=f_btn,
                             bg=ACCENT, fg="white", cursor="hand2",
                             padx=8, pady=4)
        print_btn.pack(side="right", padx=20)
        print_btn.bind("<Button-1>", lambda e: self._print_picked())

        clear_btn = tk.Label(top, text="  ✕ CLEAR  ", font=f_btn,
                             bg=CARD, fg=ACCENT2, cursor="hand2",
                             padx=8, pady=4)
        clear_btn.pack(side="right", padx=4)
        clear_btn.bind("<Button-1>", lambda e: self._clear_picked())

        # ── Status bar ─────────────────────────────────────────────
        self.status_lbl = tk.Label(root, text="Click any image to select it.",
                                   font=f_small, bg=PANEL, fg=TEXT_LO,
                                   anchor="w", pady=4)
        self.status_lbl.pack(fill="x", padx=16)

        # ── Grid canvas (scrollable) ────────────────────────────────
        grid_frame = tk.Frame(root, bg=BG)
        grid_frame.pack(fill="both", expand=True, padx=16, pady=8)

        self.grid_frame = grid_frame
        self.thumb_labels = []
        self.thumb_index_labels = []

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cell = tk.Frame(grid_frame, bg=BG, padx=PAD, pady=PAD)
                cell.grid(row=r*2, column=c, sticky="nsew")

                lbl = tk.Label(cell, bg=CARD, cursor="hand2",
                               relief="flat",
                               highlightthickness=2,
                               highlightbackground=BORDER)
                lbl.pack()

                idx_lbl = tk.Label(cell, text="", font=f_small,
                                   bg=BG, fg=TEXT_LO)
                idx_lbl.pack()

                self.thumb_labels.append(lbl)
                self.thumb_index_labels.append(idx_lbl)

        # ── Keyboard bindings ──────────────────────────────────────
        root.bind("<Left>",  lambda e: self._change_page(-1))
        root.bind("<Right>", lambda e: self._change_page(1))
        root.bind("<q>",     lambda e: root.destroy())
        root.bind("<Escape>",lambda e: root.destroy())

        # Load first category
        self._load_category(self.category)
        self._render_page()

    # ── Data loading ────────────────────────────────────────────────
    def _load_category(self, category):
        path = os.path.join(DATA_PATH, category + ".npy")
        try:
            self.raw_data = np.load(path)   # uint8 (N, 784)
            n_pages = max(1, (len(self.raw_data) + PER_PAGE - 1) // PER_PAGE)
            self.total_pages_lbl.config(text=f"/ {n_pages}")
            self.status_lbl.config(
                text=f"Loaded '{category}' — {len(self.raw_data)} samples. "
                     f"Click an image to pick it.",
                fg=TEXT_MID)
        except FileNotFoundError:
            self.raw_data = None
            self.status_lbl.config(
                text=f"File not found: {path}", fg=ACCENT2)

    # ── Rendering ────────────────────────────────────────────────────
    def _render_page(self):
        self._thumbs.clear()
        start = self.page * PER_PAGE

        for slot, lbl in enumerate(self.thumb_labels):
            idx = start + slot
            idx_lbl = self.thumb_index_labels[slot]

            if self.raw_data is None or idx >= len(self.raw_data):
                lbl.config(image="", bg=BG,
                           highlightbackground=BG)
                idx_lbl.config(text="")
                lbl.unbind("<Button-1>")
                continue

            pixels = self.raw_data[idx].reshape(28, 28)
            img = Image.fromarray(pixels, mode="L").resize(
                (THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._thumbs.append(photo)

            # Highlight if already picked
            picked_indices = self.picked.get(self.category, [])
            border_col = SUCCESS if idx in picked_indices else BORDER

            lbl.config(image=photo,
                       highlightbackground=border_col)
            idx_lbl.config(text=str(idx), fg=TEXT_LO)

            lbl.bind("<Button-1>",
                     lambda e, i=idx: self._toggle_pick(i))

        self.page_var.set(str(self.page + 1))

    # ── Interaction ──────────────────────────────────────────────────
    def _toggle_pick(self, idx):
        cat = self.category
        if cat not in self.picked:
            self.picked[cat] = []

        if idx in self.picked[cat]:
            self.picked[cat].remove(idx)
            action = "removed"
            color = WARNING
        else:
            self.picked[cat].append(idx)
            action = "picked"
            color = SUCCESS

        n = len(self.picked[cat])
        self.status_lbl.config(
            text=f"Index {idx} {action} for '{cat}'.  "
                 f"Total picked for this category: {n}",
            fg=color)
        self._render_page()   # refresh borders

    def _change_page(self, delta):
        if self.raw_data is None:
            return
        n_pages = max(1, (len(self.raw_data) + PER_PAGE - 1) // PER_PAGE)
        self.page = max(0, min(n_pages - 1, self.page + delta))
        self._render_page()

    def _on_page_entry(self, event):
        try:
            p = int(self.page_var.get()) - 1
            n_pages = max(1, (len(self.raw_data) + PER_PAGE - 1) // PER_PAGE)
            self.page = max(0, min(n_pages - 1, p))
            self._render_page()
        except ValueError:
            pass

    def _on_category_change(self, value):
        self.category = value
        self.page = 0
        self._load_category(value)
        self._render_page()

    def _print_picked(self):
        if not self.picked:
            print("\n# No images picked yet.\n")
            return

        print("\n# ── Paste this into main.py inside train_model() ──────────────")
        print("PICKED_SAMPLES = {")
        for cat in CATEGORIES:
            if cat in self.picked and self.picked[cat]:
                indices = sorted(self.picked[cat])
                print(f'    "{cat}": {indices},')
        print("}")
        print("# ────────────────────────────────────────────────────────────────\n")
        self.status_lbl.config(
            text="PICKED_SAMPLES printed to console — paste it into main.py",
            fg=SUCCESS)

    def _clear_picked(self):
        self.picked.pop(self.category, None)
        self.status_lbl.config(
            text=f"Cleared picks for '{self.category}'.", fg=WARNING)
        self._render_page()


if __name__ == "__main__":
    root = tk.Tk()
    app = BrowseApp(root)
    root.mainloop()