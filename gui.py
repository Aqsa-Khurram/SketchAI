import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import numpy as np
import math
import time
import random

from preprocess import preprocess_image


# ═══════════════════════════════════════════════════════════════
# Theme
# ═══════════════════════════════════════════════════════════════
BG          = "#07070f"
BG_RGB      = (7, 7, 15)     # BG as RGB tuple for PIL
SURFACE     = "#0e0e1c"
PANEL       = "#121220"
CARD        = "#1a1a2e"
BORDER      = "#2a2a4a"
ACCENT      = "#6c63ff"
ACCENT2     = "#ff6584"
ACCENT3     = "#43e8d8"
TEXT_HI     = "#f0eeff"
TEXT_MID    = "#a09cc0"
TEXT_LO     = "#4a4870"
SUCCESS     = "#43e8a0"
WARNING     = "#ffd166"
DANGER      = "#ff6584"

COLOR_PALETTE = [
    ("#1a1a2e", "Black"),
    ("#4361ee", "Blue"),
    ("#e63946", "Red"),
    ("#2dc653", "Green"),
    ("#f4a261", "Orange"),
    ("#c77dff", "Purple"),
    ("#48cae4", "Cyan"),
    ("#ff6b9d", "Pink"),
    ("#ffffff", "White"),
]

BRUSH_SIZES = [
    (3,  "XS"),
    (6,  "S"),
    (10, "M"),
    (16, "L"),
    (24, "XL"),
]

EMOJI_MAP = {
    "circle": "⭕", "square": "🟥", "triangle": "🔺", "tree": "🌲",
    "house": "🏠", "star": "⭐", "cloud": "☁️", "moon": "🌙",
    "sun": "☀️", "book": "📖", "beach": "🏖️", "clock": "🕐",
    "fish": "🐟", "airplane": "✈️", "leaf": "🍃", "calendar": "📅",
    "pencil": "✏️", "guitar": "🎸", "cup": "☕", "flower": "🌸",
    "cat": "🐱", "car": "🚗", "umbrella": "☂️", "bicycle": "🚲",
    "ladder": "🪜", "key": "🔑", "dog": "🐶", "lightning": "⚡",
    "chair": "🪑", "hat": "🎩",
}

# ── In-memory leaderboard (top 3 scores, persists for the session) ──
# Each entry: {"score": int, "time": str}
AI_DRAWS_LEADERBOARD: list = []
TIMER_SECONDS = 30


# ═══════════════════════════════════════════════════════════════
# Particle system (floating dots on background)
# ═══════════════════════════════════════════════════════════════
class Particle:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.x   = random.uniform(0, self.w)
        self.y   = random.uniform(0, self.h)
        self.vx  = random.uniform(-0.3, 0.3)
        self.vy  = random.uniform(-0.4, -0.1)
        self.r   = random.uniform(1.5, 3.5)
        self.a   = random.uniform(0.2, 0.7)
        self.life = random.uniform(0, 1)

    def update(self):
        self.x   += self.vx
        self.y   += self.vy
        self.life += 0.004
        if self.life > 1 or self.y < -10:
            self.reset()
            self.y = self.h + 5


# ═══════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════
class SketchApp:
    def __init__(self, root, predict_fn, on_switch_mode=None):
        self.root            = root
        self.predict_fn      = predict_fn
        self.on_switch_mode  = on_switch_mode

        self.root.title("Quick Draw AI")
        self.root.configure(bg=BG)

        # Fullscreen
        self.root.state("zoomed")          # Windows maximized
        try:
            self.root.attributes("-zoomed", True)   # Linux
        except Exception:
            pass

        self.root.update_idletasks()
        self.W = self.root.winfo_screenwidth()
        self.H = self.root.winfo_screenheight()
        self.root.geometry(f"{self.W}x{self.H}")

        # Drawing state
        self.brush_color  = "#1a1a2e"
        self.brush_radius = 6
        self.is_eraser    = False
        self.last_x       = None
        self.last_y       = None
        self.stroke_count = 0
        self.history      = []          # list of (label, emoji, conf)
        self.is_predicting = False

        # Animation state
        self.particles    = [Particle(self.W, self.H) for _ in range(60)]
        self.glow_phase   = 0.0
        self.pulse_phase  = 0.0
        self.anim_running = True

        self._load_fonts()
        self._build_ui()
        self._animate()

    # ── Fonts ──────────────────────────────────────────────────────
    def _load_fonts(self):
        self.f_title   = tkfont.Font(family="Georgia",      size=22, weight="bold")
        self.f_sub     = tkfont.Font(family="Courier New",  size=9)
        self.f_label   = tkfont.Font(family="Courier New",  size=10)
        self.f_btn     = tkfont.Font(family="Courier New",  size=10, weight="bold")
        self.f_pred    = tkfont.Font(family="Georgia",      size=28, weight="bold")
        self.f_pred_sm = tkfont.Font(family="Georgia",      size=14, weight="bold")
        self.f_conf    = tkfont.Font(family="Courier New",  size=9)
        self.f_emoji   = tkfont.Font(family="Segoe UI Emoji", size=38)
        self.f_hist    = tkfont.Font(family="Courier New",  size=9)
        self.f_sec     = tkfont.Font(family="Courier New",  size=7)

    # ── UI Build ───────────────────────────────────────────────────
    def _build_ui(self):
        # Background canvas (particles)
        self.bg_canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # ── Three-column layout ──
        LEFT_W  = 220
        RIGHT_W = 300
        MID_W   = self.W - LEFT_W - RIGHT_W

        # Left sidebar
        self.left = tk.Frame(self.root, bg=PANEL, width=LEFT_W)
        self.left.place(x=0, y=0, width=LEFT_W, relheight=1)

        # Center area
        self.center = tk.Frame(self.root, bg=BG)
        self.center.place(x=LEFT_W, y=0, width=MID_W, relheight=1)

        # Right sidebar
        self.right = tk.Frame(self.root, bg=PANEL, width=RIGHT_W)
        self.right.place(x=LEFT_W + MID_W, y=0, width=RIGHT_W, relheight=1)

        self._build_left_sidebar(LEFT_W)
        self._build_center(MID_W)
        self._build_right_sidebar(RIGHT_W)

    # ── Left sidebar: branding + tools ────────────────────────────
    def _build_left_sidebar(self, w):
        # Gradient-like top block
        top = tk.Frame(self.left, bg=ACCENT, height=4)
        top.pack(fill="x")

        # Logo area
        logo_frame = tk.Frame(self.left, bg=PANEL, pady=20)
        logo_frame.pack(fill="x")
        tk.Label(logo_frame, text="✦ QUICK", font=self.f_title,
                 bg=PANEL, fg=ACCENT).pack()
        tk.Label(logo_frame, text="DRAW  AI", font=self.f_title,
                 bg=PANEL, fg=TEXT_HI).pack()
        tk.Label(logo_frame, text="sketch · classify · discover",
                 font=self.f_sub, bg=PANEL, fg=TEXT_LO).pack(pady=(4, 0))

        self._h_sep(self.left)

        # Color picker
        self._section(self.left, "BRUSH COLOR")
        color_grid = tk.Frame(self.left, bg=PANEL)
        color_grid.pack(padx=16, pady=4)
        self.color_btns = {}
        for i, (hex_c, name) in enumerate(COLOR_PALETTE):
            row, col = divmod(i, 3)
            border_color = "#ffffff" if hex_c == self.brush_color else BORDER
            btn = tk.Label(color_grid, bg=hex_c, width=3, height=1,
                           cursor="hand2",
                           highlightthickness=2,
                           highlightbackground=border_color)
            btn.grid(row=row, column=col, padx=3, pady=3)
            btn.bind("<Button-1>", lambda e, c=hex_c: self._select_color(c))
            self.color_btns[hex_c] = btn

        self._h_sep(self.left)

        # Brush size
        self._section(self.left, "BRUSH SIZE")
        size_frame = tk.Frame(self.left, bg=PANEL)
        size_frame.pack(padx=16, pady=4, fill="x")
        self.size_btns = {}
        for radius, label in BRUSH_SIZES:
            is_sel = radius == self.brush_radius
            btn = tk.Label(size_frame,
                           text=f"  {label}  ",
                           font=self.f_btn,
                           bg=ACCENT if is_sel else CARD,
                           fg=TEXT_HI if is_sel else TEXT_MID,
                           padx=6, pady=4,
                           cursor="hand2")
            btn.pack(side="left", padx=2)
            btn.bind("<Button-1>", lambda e, r=radius: self._select_size(r))
            self.size_btns[radius] = btn

        self._h_sep(self.left)

        # Tools
        self._section(self.left, "TOOLS")
        tool_frame = tk.Frame(self.left, bg=PANEL)
        tool_frame.pack(padx=16, pady=4, fill="x")

        self.eraser_btn = self._tool_btn(tool_frame, "⌫  ERASER", self._toggle_eraser)
        self.eraser_btn.pack(fill="x", pady=2)

        clear_btn = self._tool_btn(tool_frame, "⟳  CLEAR CANVAS", self._clear,
                                   fg=DANGER)
        clear_btn.pack(fill="x", pady=2)

        if self.on_switch_mode:
            switch_btn = self._tool_btn(tool_frame, "⇄  SWITCH MODE",
                                        self.on_switch_mode, fg=ACCENT3)
            switch_btn.pack(fill="x", pady=2)

        self._h_sep(self.left)

        # Stroke counter
        self._section(self.left, "SESSION STATS")
        stats_frame = tk.Frame(self.left, bg=PANEL)
        stats_frame.pack(padx=16, pady=4, fill="x")

        self.stroke_var = tk.StringVar(value="0")
        self.guess_var  = tk.StringVar(value="0")

        for label, var in [("Strokes", self.stroke_var), ("Guesses", self.guess_var)]:
            row = tk.Frame(stats_frame, bg=CARD, pady=6)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=self.f_conf,
                     bg=CARD, fg=TEXT_MID).pack()
            tk.Label(row, textvariable=var, font=self.f_pred_sm,
                     bg=CARD, fg=ACCENT).pack()

    # ── Center: canvas ─────────────────────────────────────────────
    def _build_center(self, mid_w):
        # Top label
        header = tk.Frame(self.center, bg=BG, pady=16)
        header.pack(fill="x")
        tk.Label(header, text="Draw anything!", font=self.f_pred_sm,
                 bg=BG, fg=TEXT_MID).pack()

        # Canvas wrapper with glowing border (achieved with a Frame + Canvas)
        canvas_h = min(self.H - 180, 680)
        canvas_w = min(mid_w - 60, canvas_h)
        self.canvas_size = canvas_w

        self.glow_frame = tk.Frame(self.center, bg=ACCENT,
                                   padx=3, pady=3)
        self.glow_frame.pack(padx=20)

        canvas_bg_frame = tk.Frame(self.glow_frame, bg="#ffffff")
        canvas_bg_frame.pack()

        self.canvas = tk.Canvas(canvas_bg_frame,
                                width=canvas_w, height=canvas_w,
                                bg="#ffffff", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()

        self.pil_image = Image.new("RGB", (canvas_w, canvas_w), "white")
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Bottom: predict button + status
        bottom = tk.Frame(self.center, bg=BG, pady=12)
        bottom.pack(fill="x")

        self.predict_btn = tk.Label(bottom,
                                    text="  ▶  PREDICT  ",
                                    font=self.f_btn,
                                    bg=ACCENT, fg="white",
                                    padx=24, pady=10,
                                    cursor="hand2")
        self.predict_btn.pack()
        self.predict_btn.bind("<Button-1>",  lambda e: self._predict())
        self.predict_btn.bind("<Enter>",     lambda e: self.predict_btn.config(bg="#8b84ff"))
        self.predict_btn.bind("<Leave>",     lambda e: self.predict_btn.config(bg=ACCENT))

        self.status_lbl = tk.Label(bottom, text="ready to draw",
                                   font=self.f_sub, bg=BG, fg=TEXT_LO)
        self.status_lbl.pack(pady=(4, 0))

    # ── Right sidebar: predictions + history ──────────────────────
    def _build_right_sidebar(self, w):
        top = tk.Frame(self.right, bg=ACCENT2, height=4)
        top.pack(fill="x")

        # Prediction display
        pred_area = tk.Frame(self.right, bg=PANEL, pady=16)
        pred_area.pack(fill="x")

        self._section(pred_area, "BEST GUESS")

        # Emoji
        self.emoji_lbl = tk.Label(pred_area, text="?",
                                  font=self.f_emoji,
                                  bg=PANEL, fg=TEXT_HI)
        self.emoji_lbl.pack(pady=(4, 0))

        # Top prediction name
        self.top_pred_lbl = tk.Label(pred_area, text="—",
                                     font=self.f_pred,
                                     bg=PANEL, fg=ACCENT,
                                     wraplength=w - 30)
        self.top_pred_lbl.pack()

        # Confidence bar
        self.conf_bar_canvas = tk.Canvas(pred_area, width=w - 40, height=10,
                                          bg=CARD, highlightthickness=0)
        self.conf_bar_canvas.pack(pady=(2, 0))

        self.top_conf_lbl = tk.Label(pred_area, text="",
                                     font=self.f_conf,
                                     bg=PANEL, fg=SUCCESS)
        self.top_conf_lbl.pack()

        self._h_sep(self.right)

        # Alt predictions
        self._section(self.right, "ALSO MAYBE")
        self.alt_frames = []
        for _ in range(2):
            alt_row = tk.Frame(self.right, bg=CARD,
                               highlightthickness=1,
                               highlightbackground=BORDER)
            alt_row.pack(fill="x", padx=14, pady=3)

            inner = tk.Frame(alt_row, bg=CARD)
            inner.pack(fill="x", padx=10, pady=6)

            lbl  = tk.Label(inner, text="—", font=self.f_pred_sm,
                            bg=CARD, fg=TEXT_MID)
            lbl.pack(side="left")

            conf = tk.Label(inner, text="", font=self.f_conf,
                            bg=CARD, fg=TEXT_LO)
            conf.pack(side="right")

            bar  = tk.Canvas(alt_row, height=3, bg=SURFACE,
                             highlightthickness=0)
            bar.pack(fill="x")

            self.alt_frames.append((lbl, conf, bar))

        self._h_sep(self.right)

        # History
        self._section(self.right, "HISTORY")
        self.history_frame = tk.Frame(self.right, bg=PANEL)
        self.history_frame.pack(fill="both", expand=True, padx=14, pady=4)

        self.history_items = []
        for _ in range(6):
            item = tk.Label(self.history_frame, text="",
                            font=self.f_hist,
                            bg=PANEL, fg=TEXT_LO,
                            anchor="w")
            item.pack(fill="x", pady=1)
            self.history_items.append(item)

    # ── Helpers ────────────────────────────────────────────────────
    def _h_sep(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=8)

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=self.f_sec,
                 bg=parent.cget("bg"), fg=TEXT_LO,
                 anchor="w").pack(fill="x", padx=16, pady=(4, 2))

    def _tool_btn(self, parent, text, cmd, fg=TEXT_MID):
        btn = tk.Label(parent, text=text, font=self.f_label,
                       bg=CARD, fg=fg, padx=10, pady=6,
                       anchor="w", cursor="hand2")
        btn.bind("<Button-1>", lambda e: cmd())
        btn.bind("<Enter>",    lambda e: btn.config(bg=BORDER))
        btn.bind("<Leave>",    lambda e: btn.config(bg=CARD))
        return btn

    # ── Color / size / eraser ──────────────────────────────────────
    def _select_color(self, color):
        if self.is_eraser:
            self.is_eraser = False
            self.eraser_btn.config(bg=CARD, fg=TEXT_MID)
        for c, btn in self.color_btns.items():
            btn.config(highlightbackground="white" if c == color else BORDER)
        self.brush_color = color

    def _select_size(self, radius):
        for r, btn in self.size_btns.items():
            btn.config(bg=ACCENT if r == radius else CARD,
                       fg=TEXT_HI if r == radius else TEXT_MID)
        self.brush_radius = radius

    def _toggle_eraser(self):
        self.is_eraser = not self.is_eraser
        if self.is_eraser:
            self.eraser_btn.config(bg=ACCENT2, fg=TEXT_HI)
            for btn in self.color_btns.values():
                btn.config(highlightbackground=BORDER)
        else:
            self.eraser_btn.config(bg=CARD, fg=TEXT_MID)

    # ── Drawing ────────────────────────────────────────────────────
    def _on_press(self, event):
        self.last_x, self.last_y = event.x, event.y
        self._dot(event.x, event.y)

    def _on_drag(self, event):
        if self.last_x is not None:
            self._line(self.last_x, self.last_y, event.x, event.y)
        self.last_x, self.last_y = event.x, event.y

    def _on_release(self, event):
        self.last_x = self.last_y = None
        self.stroke_count += 1
        self.stroke_var.set(str(self.stroke_count))
        self._predict()

    def _color_for_draw(self):
        return "white" if self.is_eraser else self.brush_color

    def _radius_for_draw(self):
        return self.brush_radius * 2 if self.is_eraser else self.brush_radius

    def _dot(self, x, y):
        c = self._color_for_draw()
        r = self._radius_for_draw()
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=c, outline="")
        self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=c)

    def _line(self, x1, y1, x2, y2):
        dist  = math.hypot(x2-x1, y2-y1)
        steps = max(1, int(dist / 2))
        for i in range(steps + 1):
            t  = i / steps
            xi = int(x1 + t*(x2-x1))
            yi = int(y1 + t*(y2-y1))
            self._dot(xi, yi)

    # ── Predict ────────────────────────────────────────────────────
    def _predict(self):
        if self.is_predicting:
            return
        self.is_predicting = True
        self.status_lbl.config(text="🤔  thinking…", fg=WARNING)
        self.root.update_idletasks()

        try:
            img    = np.array(self.pil_image)
            vector = preprocess_image(img)
            result = self.predict_fn(vector)
            self._display_result(result)
        except Exception as ex:
            self.status_lbl.config(text=f"Error: {ex}", fg=DANGER)
        finally:
            self.is_predicting = False

    def _display_result(self, result):
        lines = result.strip().split("\n")

        def parse(line, rank):
            line = line.replace(f"#{rank}", "").strip()
            parts = line.split("(")
            name = parts[0].strip()
            conf_str = parts[1].replace(")", "").strip() if len(parts) > 1 else "0%"
            conf_val = float(conf_str.replace("%", "")) / 100.0
            return name, conf_str, conf_val

        if lines:
            name, conf_str, conf_val = parse(lines[0], 1)
            emoji = EMOJI_MAP.get(name.lower(), "🎨")
            self.emoji_lbl.config(text=emoji)
            self.top_pred_lbl.config(text=name.upper())
            self.top_conf_lbl.config(text=f"confidence  {conf_str}")
            self._draw_conf_bar(self.conf_bar_canvas, conf_val, ACCENT)

            # Add to history
            self.history.insert(0, (name, emoji, conf_str))
            self.history = self.history[:6]
            self.guess_var.set(str(len(self.history)))
            self._refresh_history()

        for i, (lbl, conf_lbl, bar) in enumerate(self.alt_frames):
            if i + 1 < len(lines):
                name, conf_str, conf_val = parse(lines[i+1], i+2)
                emoji = EMOJI_MAP.get(name.lower(), "")
                lbl.config(text=f"{emoji}  {name}")
                conf_lbl.config(text=conf_str)
                self._draw_conf_bar(bar, conf_val, ACCENT3)
            else:
                lbl.config(text="—")
                conf_lbl.config(text="")
                bar.delete("all")

        self.status_lbl.config(text="done ✓", fg=SUCCESS)

    def _draw_conf_bar(self, bar_canvas, value, color):
        bar_canvas.delete("all")
        w = bar_canvas.winfo_width()
        h = bar_canvas.winfo_height()
        if w < 2:
            return
        filled = int(w * max(0, min(1, value)))
        bar_canvas.create_rectangle(0, 0, w, h, fill=SURFACE, outline="")
        if filled > 0:
            bar_canvas.create_rectangle(0, 0, filled, h, fill=color, outline="")

    def _refresh_history(self):
        for i, item_lbl in enumerate(self.history_items):
            if i < len(self.history):
                name, emoji, conf = self.history[i]
                item_lbl.config(
                    text=f"  {emoji}  {name:<12s}  {conf}",
                    fg=TEXT_MID if i == 0 else TEXT_LO
                )
            else:
                item_lbl.config(text="")

    # ── Clear ──────────────────────────────────────────────────────
    def _clear(self):
        self.canvas.delete("all")
        self.pil_draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill="white")
        self.stroke_count = 0
        self.stroke_var.set("0")
        self.emoji_lbl.config(text="?")
        self.top_pred_lbl.config(text="—")
        self.top_conf_lbl.config(text="")
        self.conf_bar_canvas.delete("all")
        for lbl, conf, bar in self.alt_frames:
            lbl.config(text="—")
            conf.config(text="")
            bar.delete("all")
        self.status_lbl.config(text="canvas cleared", fg=TEXT_LO)

    # ── Background animation ───────────────────────────────────────
    def _animate(self):
        if not self.anim_running:
            return

        self.glow_phase  = (self.glow_phase  + 0.03) % (2 * math.pi)
        self.pulse_phase = (self.pulse_phase + 0.02) % (2 * math.pi)

        # Glow border on canvas frame
        r = int(80 + 40 * math.sin(self.glow_phase))
        g = int(50 + 30 * math.sin(self.glow_phase + 2))
        b = int(220 + 35 * math.sin(self.glow_phase + 4))
        glow_color = f"#{r:02x}{g:02x}{b:02x}"
        self.glow_frame.config(bg=glow_color)

        # Draw particles on bg canvas
        self.bg_canvas.delete("particle")
        for p in self.particles:
            p.update()
            alpha_hex = format(int(p.a * 60), '02x')
            col = f"#3{alpha_hex}3{alpha_hex}ff"
            x, y, r = p.x, p.y, p.r
            try:
                self.bg_canvas.create_oval(
                    x-r, y-r, x+r, y+r,
                    fill=col, outline="", tags="particle"
                )
            except Exception:
                pass

        self.root.after(40, self._animate)   # ~25 fps

    def _on_close(self):
        self.anim_running = False
        self.root.destroy()

# ═══════════════════════════════════════════════════════════════
# Mode Selection Screen
# ═══════════════════════════════════════════════════════════════
class ModeSelectApp:
    """
    Splash screen shown at startup.  The player chooses who draws.
    """
    def __init__(self, root, on_human_draws, on_ai_draws):
        self.root = root
        self.root.configure(bg=BG)
        self.root.title("Quick Draw AI – Choose Mode")

        # Particles
        self.root.update_idletasks()
        self.W = self.root.winfo_screenwidth()
        self.H = self.root.winfo_screenheight()
        self.root.geometry(f"{self.W}x{self.H}")
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        self.particles   = [Particle(self.W, self.H) for _ in range(80)]
        self.anim_running = True
        self.glow_phase  = 0.0

        f_title = tkfont.Font(family="Georgia",     size=36, weight="bold")
        f_sub   = tkfont.Font(family="Courier New", size=11)
        f_btn   = tkfont.Font(family="Georgia",     size=20, weight="bold")
        f_desc  = tkfont.Font(family="Courier New", size=10)

        # Background canvas
        self.bg_canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # Content frame (centered)
        frame = tk.Frame(root, bg=BG)
        frame.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(frame, text="✦ QUICK DRAW AI", font=f_title,
                 bg=BG, fg=ACCENT).pack(pady=(0, 4))
        tk.Label(frame, text="choose your challenge",
                 font=f_sub, bg=BG, fg=TEXT_LO).pack(pady=(0, 40))

        # ── Card: Human draws ──────────────────────────────────
        card1 = tk.Frame(frame, bg=CARD,
                         highlightthickness=2,
                         highlightbackground=ACCENT,
                         padx=40, pady=28,
                         cursor="hand2")
        card1.pack(pady=10, fill="x")

        tk.Label(card1, text="✏️  YOU DRAW", font=f_btn,
                 bg=CARD, fg=ACCENT).pack()
        tk.Label(card1, text="Draw a sketch — the AI tries to guess it",
                 font=f_desc, bg=CARD, fg=TEXT_MID).pack(pady=(6, 0))

        for widget in [card1] + card1.winfo_children():
            widget.bind("<Button-1>", lambda e: on_human_draws())

        card1.bind("<Enter>", lambda e: card1.config(highlightbackground=ACCENT3))
        card1.bind("<Leave>", lambda e: card1.config(highlightbackground=ACCENT))

        # ── Card: AI draws ─────────────────────────────────────
        card2 = tk.Frame(frame, bg=CARD,
                         highlightthickness=2,
                         highlightbackground=ACCENT2,
                         padx=40, pady=28,
                         cursor="hand2")
        card2.pack(pady=10, fill="x")

        tk.Label(card2, text="🤖  AI DRAWS", font=f_btn,
                 bg=CARD, fg=ACCENT2).pack()
        tk.Label(card2, text="The AI draws a sketch — you try to guess it",
                 font=f_desc, bg=CARD, fg=TEXT_MID).pack(pady=(6, 0))

        for widget in [card2] + card2.winfo_children():
            widget.bind("<Button-1>", lambda e: on_ai_draws())

        card2.bind("<Enter>", lambda e: card2.config(highlightbackground=WARNING))
        card2.bind("<Leave>", lambda e: card2.config(highlightbackground=ACCENT2))

        self._animate()

    def _animate(self):
        if not self.anim_running:
            return
        self.glow_phase = (self.glow_phase + 0.03) % (2 * math.pi)
        self.bg_canvas.delete("particle")
        for p in self.particles:
            p.update()
            alpha_hex = format(int(p.a * 60), '02x')
            col = f"#3{alpha_hex}3{alpha_hex}ff"
            x, y, r = p.x, p.y, p.r
            try:
                self.bg_canvas.create_oval(x-r, y-r, x+r, y+r,
                                           fill=col, outline="", tags="particle")
            except Exception:
                pass
        self.root.after(40, self._animate)

    def destroy(self):
        self.anim_running = False
        for w in self.root.winfo_children():
            w.destroy()


# ═══════════════════════════════════════════════════════════════
# Timer Mode Selection Screen (shown before AI Draws)
# ═══════════════════════════════════════════════════════════════
class TimerModeSelectApp:
    """
    Shown when the player picks 'AI Draws'.
    Offers: 30-second Blitz or Relaxed (no timer).
    Leaderboard for timed mode is shown here.
    """
    def __init__(self, root, on_timed, on_relaxed, on_back):
        self.root         = root
        self.anim_running = True
        self.glow_phase   = 0.0

        self.root.configure(bg=BG)
        self.root.title("Quick Draw AI – AI Draws Mode")
        self.root.update_idletasks()
        self.W = self.root.winfo_screenwidth()
        self.H = self.root.winfo_screenheight()
        self.root.geometry(f"{self.W}x{self.H}")
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        self.particles = [Particle(self.W, self.H) for _ in range(80)]

        f_title  = tkfont.Font(family="Georgia",     size=30, weight="bold")
        f_sub    = tkfont.Font(family="Courier New", size=10)
        f_btn    = tkfont.Font(family="Georgia",     size=18, weight="bold")
        f_desc   = tkfont.Font(family="Courier New", size=10)
        f_lb_hdr = tkfont.Font(family="Courier New", size=9,  weight="bold")
        f_lb_row = tkfont.Font(family="Courier New", size=11)

        self.bg_canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # ── Outer wrapper ──────────────────────────────────────
        outer = tk.Frame(root, bg=BG)
        outer.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(outer, text="🤖  AI DRAWS — Choose Mode", font=f_title,
                 bg=BG, fg=ACCENT2).pack(pady=(0, 4))
        tk.Label(outer, text="timed blitz or relaxed play?",
                 font=f_sub, bg=BG, fg=TEXT_LO).pack(pady=(0, 30))

        cards_row = tk.Frame(outer, bg=BG)
        cards_row.pack()

        # ── Card: Timed ────────────────────────────────────────
        c_timed = tk.Frame(cards_row, bg=CARD,
                           highlightthickness=2,
                           highlightbackground=WARNING,
                           padx=34, pady=24, cursor="hand2")
        c_timed.pack(side="left", padx=16)

        tk.Label(c_timed, text=f"⏱  {TIMER_SECONDS}s BLITZ", font=f_btn,
                 bg=CARD, fg=WARNING).pack()
        tk.Label(c_timed,
                 text=f"Guess as many as you can\nin {TIMER_SECONDS} seconds!",
                 font=f_desc, bg=CARD, fg=TEXT_MID, justify="center").pack(pady=(6, 0))

        for w in [c_timed] + c_timed.winfo_children():
            w.bind("<Button-1>", lambda e: on_timed())
        c_timed.bind("<Enter>", lambda e: c_timed.config(highlightbackground=ACCENT3))
        c_timed.bind("<Leave>", lambda e: c_timed.config(highlightbackground=WARNING))

        # ── Card: Relaxed ──────────────────────────────────────
        c_relax = tk.Frame(cards_row, bg=CARD,
                           highlightthickness=2,
                           highlightbackground=ACCENT3,
                           padx=34, pady=24, cursor="hand2")
        c_relax.pack(side="left", padx=16)

        tk.Label(c_relax, text="😌  RELAXED", font=f_btn,
                 bg=CARD, fg=ACCENT3).pack()
        tk.Label(c_relax,
                 text="No timer — guess at\nyour own pace",
                 font=f_desc, bg=CARD, fg=TEXT_MID, justify="center").pack(pady=(6, 0))

        for w in [c_relax] + c_relax.winfo_children():
            w.bind("<Button-1>", lambda e: on_relaxed())
        c_relax.bind("<Enter>", lambda e: c_relax.config(highlightbackground=ACCENT))
        c_relax.bind("<Leave>", lambda e: c_relax.config(highlightbackground=ACCENT3))

        # ── Back button ────────────────────────────────────────
        back_btn = tk.Label(outer, text="← BACK",
                            font=f_desc, bg=BG, fg=TEXT_LO, cursor="hand2")
        back_btn.pack(pady=(18, 0))
        back_btn.bind("<Button-1>", lambda e: on_back())
        back_btn.bind("<Enter>",    lambda e: back_btn.config(fg=TEXT_MID))
        back_btn.bind("<Leave>",    lambda e: back_btn.config(fg=TEXT_LO))

        # ── Leaderboard ────────────────────────────────────────
        if AI_DRAWS_LEADERBOARD:
            self._h_sep(outer)
            tk.Label(outer, text="🏆  TOP SCORES  (30s Blitz)",
                     font=f_lb_hdr, bg=BG, fg=WARNING).pack(pady=(8, 4))

            medals = ["🥇", "🥈", "🥉"]
            for i, entry in enumerate(AI_DRAWS_LEADERBOARD[:3]):
                medal = medals[i] if i < len(medals) else "  "
                row_frame = tk.Frame(outer, bg=CARD, padx=20, pady=6,
                                     highlightthickness=1,
                                     highlightbackground=BORDER)
                row_frame.pack(fill="x", pady=2, padx=40)
                colors = [WARNING, TEXT_MID, ACCENT3]
                fg_c   = colors[i] if i < len(colors) else TEXT_LO
                tk.Label(row_frame,
                         text=f"{medal}  {entry['score']} correct   —   {entry['date']}",
                         font=f_lb_row, bg=CARD, fg=fg_c).pack()

        self._animate()

    def _h_sep(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=10)

    def _animate(self):
        if not self.anim_running:
            return
        self.glow_phase = (self.glow_phase + 0.03) % (2 * math.pi)
        self.bg_canvas.delete("particle")
        for p in self.particles:
            p.update()
            alpha_hex = format(int(p.a * 60), '02x')
            col = f"#3{alpha_hex}3{alpha_hex}ff"
            x, y, r = p.x, p.y, p.r
            try:
                self.bg_canvas.create_oval(x-r, y-r, x+r, y+r,
                                           fill=col, outline="", tags="particle")
            except Exception:
                pass
        self.root.after(40, self._animate)

    def destroy(self):
        self.anim_running = False
        for w in self.root.winfo_children():
            w.destroy()


# ═══════════════════════════════════════════════════════════════
# AI Draws – Human Guesses
# ═══════════════════════════════════════════════════════════════
class AIDrawApp:
    """
    The AI picks a secret category, renders a pixel drawing from a
    prototype centroid (upscaled to canvas size), and the human
    types guesses.  Correct = green flash; wrong = red flash.
    """

    REVEAL_STEPS   = 40          # how many 'strokes' to reveal the drawing
    REVEAL_DELAY   = 40          # ms between strokes (total ~1.6 s)
    HINT_THRESHOLD = 3           # wrong guesses before a hint is offered

    def __init__(self, root, categories, pixel_prototypes,
                 on_switch_mode=None, timed_mode=False):
        """
        Parameters
        ----------
        categories       : list[str]   – all possible category names
        pixel_prototypes : dict        – {label: [28x28 float arrays in [0,1]]}
                                         Raw pixel-space samples, used directly
                                         for rendering clear QuickDraw images.
        on_switch_mode   : callable | None
        timed_mode       : bool        – if True, 30-second countdown is active
        """
        self.root              = root
        self.categories        = categories
        self.pixel_prototypes  = pixel_prototypes
        self.on_switch_mode    = on_switch_mode
        self.timed_mode        = timed_mode

        self.secret_label   = None
        self.secret_pixels  = None          # flat 784 floats [0,1]
        self.wrong_guesses  = 0
        self.score          = 0
        self.round_num      = 0
        self._reveal_job    = None
        self._reveal_pts    = []

        # Timer state
        self._time_left     = TIMER_SECONDS
        self._timer_job     = None
        self._game_over     = False
        self._overlay       = None          # game-over overlay frame
        self._revealed_idx  = 0

        self.root.title("Quick Draw AI – AI Draws")
        self.root.configure(bg=BG)
        try:
            self.root.state("zoomed")
        except Exception:
            pass
        self.root.update_idletasks()
        self.W = self.root.winfo_screenwidth()
        self.H = self.root.winfo_screenheight()
        self.root.geometry(f"{self.W}x{self.H}")

        self.particles    = [Particle(self.W, self.H) for _ in range(60)]
        self.anim_running = True
        self.glow_phase   = 0.0

        self._load_fonts()
        self._build_ui()
        self._animate()
        self._new_round()

    # ── Fonts ──────────────────────────────────────────────────────
    def _load_fonts(self):
        self.f_title   = tkfont.Font(family="Georgia",      size=22, weight="bold")
        self.f_sub     = tkfont.Font(family="Courier New",  size=9)
        self.f_label   = tkfont.Font(family="Courier New",  size=10)
        self.f_btn     = tkfont.Font(family="Courier New",  size=10, weight="bold")
        self.f_pred    = tkfont.Font(family="Georgia",      size=28, weight="bold")
        self.f_pred_sm = tkfont.Font(family="Georgia",      size=14, weight="bold")
        self.f_conf    = tkfont.Font(family="Courier New",  size=9)
        self.f_emoji   = tkfont.Font(family="Segoe UI Emoji", size=38)
        self.f_sec     = tkfont.Font(family="Courier New",  size=7)
        self.f_guess   = tkfont.Font(family="Georgia",      size=16)

    # ── UI ─────────────────────────────────────────────────────────
    def _build_ui(self):
        self.bg_canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        LEFT_W  = 220
        RIGHT_W = 300
        MID_W   = self.W - LEFT_W - RIGHT_W

        # Left sidebar
        self.left = tk.Frame(self.root, bg=PANEL, width=LEFT_W)
        self.left.place(x=0, y=0, width=LEFT_W, relheight=1)

        # Center
        self.center = tk.Frame(self.root, bg=BG)
        self.center.place(x=LEFT_W, y=0, width=MID_W, relheight=1)

        # Right sidebar
        self.right = tk.Frame(self.root, bg=PANEL, width=RIGHT_W)
        self.right.place(x=LEFT_W + MID_W, y=0, width=RIGHT_W, relheight=1)

        self._build_left(LEFT_W)
        self._build_center(MID_W)
        self._build_right(RIGHT_W)

    # ── Left ───────────────────────────────────────────────────────
    def _build_left(self, w):
        tk.Frame(self.left, bg=ACCENT2, height=4).pack(fill="x")

        logo = tk.Frame(self.left, bg=PANEL, pady=20)
        logo.pack(fill="x")
        tk.Label(logo, text="✦ QUICK", font=self.f_title,
                 bg=PANEL, fg=ACCENT2).pack()
        tk.Label(logo, text="DRAW  AI", font=self.f_title,
                 bg=PANEL, fg=TEXT_HI).pack()
        mode_tag = "⏱ 30s BLITZ" if self.timed_mode else "ai draws · you guess"
        tk.Label(logo, text=mode_tag,
                 font=self.f_sub, bg=PANEL, fg=WARNING if self.timed_mode else TEXT_LO
                 ).pack(pady=(4, 0))

        self._h_sep(self.left)

        # ── Timer (timed mode only) ────────────────────────────
        if self.timed_mode:
            self._section(self.left, "TIME LEFT")
            timer_frame = tk.Frame(self.left, bg=CARD, pady=10)
            timer_frame.pack(padx=16, pady=4, fill="x")
            self.timer_var = tk.StringVar(value=str(TIMER_SECONDS))
            self.timer_lbl = tk.Label(timer_frame,
                                      textvariable=self.timer_var,
                                      font=tkfont.Font(family="Georgia", size=36, weight="bold"),
                                      bg=CARD, fg=WARNING)
            self.timer_lbl.pack()
            tk.Label(timer_frame, text="seconds", font=self.f_conf,
                     bg=CARD, fg=TEXT_LO).pack()
            self._h_sep(self.left)

        self._section(self.left, "SESSION STATS")

        stats = tk.Frame(self.left, bg=PANEL)
        stats.pack(padx=16, pady=4, fill="x")

        self.score_var = tk.StringVar(value="0")
        self.round_var = tk.StringVar(value="0")

        for label, var, color in [
            ("Score",  self.score_var, SUCCESS),
            ("Round",  self.round_var, ACCENT),
        ]:
            row = tk.Frame(stats, bg=CARD, pady=6)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=self.f_conf,
                     bg=CARD, fg=TEXT_MID).pack()
            tk.Label(row, textvariable=var, font=self.f_pred_sm,
                     bg=CARD, fg=color).pack()

        self._h_sep(self.left)
        self._section(self.left, "TOOLS")

        tool_frame = tk.Frame(self.left, bg=PANEL)
        tool_frame.pack(padx=16, pady=4, fill="x")

        skip_btn = self._tool_btn(tool_frame, "⏭  SKIP ROUND",
                                  self._skip_round, fg=WARNING)
        skip_btn.pack(fill="x", pady=2)

        hint_btn = self._tool_btn(tool_frame, "💡  HINT",
                                  self._show_hint, fg=ACCENT3)
        hint_btn.pack(fill="x", pady=2)

        if self.on_switch_mode:
            sw_btn = self._tool_btn(tool_frame, "⇄  SWITCH MODE",
                                    self.on_switch_mode, fg=ACCENT)
            sw_btn.pack(fill="x", pady=2)

    # ── Center ─────────────────────────────────────────────────────
    def _build_center(self, mid_w):
        header = tk.Frame(self.center, bg=BG, pady=16)
        header.pack(fill="x")
        self.prompt_lbl = tk.Label(header,
                                   text="What is the AI drawing?",
                                   font=self.f_pred_sm, bg=BG, fg=TEXT_MID)
        self.prompt_lbl.pack()

        canvas_h = min(self.H - 260, 300)
        canvas_w = min(mid_w - 120, canvas_h)
        self.canvas_size = canvas_w
        self.cell_size   = canvas_w // 28        # each pixel = this many screen px

        self.glow_frame = tk.Frame(self.center, bg=ACCENT2, padx=3, pady=3)
        self.glow_frame.pack(padx=20)

        inner = tk.Frame(self.glow_frame, bg=BG)
        inner.pack()

        draw_size = self.canvas_size
        self.canvas = tk.Canvas(inner,
                                width=draw_size, height=draw_size,
                                bg="#000000", highlightthickness=0)
        self.canvas.pack()

        # Guess entry area
        guess_area = tk.Frame(self.center, bg=BG, pady=14)
        guess_area.pack(fill="x")

        self.guess_var_entry = tk.StringVar()
        entry = tk.Entry(guess_area,
                         textvariable=self.guess_var_entry,
                         font=self.f_guess,
                         bg=CARD, fg=TEXT_HI,
                         insertbackground=ACCENT,
                         relief="flat",
                         justify="center",
                         width=20)
        entry.pack(pady=(0, 6))
        entry.bind("<Return>", lambda e: self._check_guess())
        entry.focus_set()
        self.entry = entry

        submit_btn = tk.Label(guess_area,
                              text="  ▶  GUESS  ",
                              font=self.f_btn,
                              bg=ACCENT2, fg="white",
                              padx=20, pady=8,
                              cursor="hand2")
        submit_btn.pack()
        submit_btn.bind("<Button-1>", lambda e: self._check_guess())
        submit_btn.bind("<Enter>",    lambda e: submit_btn.config(bg="#ff8fa3"))
        submit_btn.bind("<Leave>",    lambda e: submit_btn.config(bg=ACCENT2))

        self.status_lbl = tk.Label(guess_area, text="",
                                   font=self.f_sub, bg=BG, fg=TEXT_LO)
        self.status_lbl.pack(pady=(4, 0))

    # ── Right ──────────────────────────────────────────────────────
    def _build_right(self, w):
        tk.Frame(self.right, bg=ACCENT, height=4).pack(fill="x")

        result_area = tk.Frame(self.right, bg=PANEL, pady=16)
        result_area.pack(fill="x")

        self._section(result_area, "RESULT")

        self.result_emoji = tk.Label(result_area, text="❓",
                                     font=self.f_emoji, bg=PANEL, fg=TEXT_HI)
        self.result_emoji.pack(pady=(4, 0))

        self.result_lbl = tk.Label(result_area, text="—",
                                   font=self.f_pred, bg=PANEL, fg=ACCENT,
                                   wraplength=w - 30)
        self.result_lbl.pack()

        self.result_sub = tk.Label(result_area, text="",
                                   font=self.f_conf, bg=PANEL, fg=TEXT_MID)
        self.result_sub.pack()

        self._h_sep(self.right)
        self._section(self.right, "GUESS HISTORY")

        self.hist_frame = tk.Frame(self.right, bg=PANEL)
        self.hist_frame.pack(fill="both", expand=True, padx=14, pady=4)

        self.hist_labels = []
        for _ in range(8):
            lbl = tk.Label(self.hist_frame, text="", font=self.f_label,
                           bg=PANEL, fg=TEXT_LO, anchor="w")
            lbl.pack(fill="x", pady=1)
            self.hist_labels.append(lbl)

        self.guess_history = []

    # ── Helpers ────────────────────────────────────────────────────
    def _h_sep(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=8)

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=self.f_sec,
                 bg=parent.cget("bg"), fg=TEXT_LO,
                 anchor="w").pack(fill="x", padx=16, pady=(4, 2))

    def _tool_btn(self, parent, text, cmd, fg=TEXT_MID):
        btn = tk.Label(parent, text=text, font=self.f_label,
                       bg=CARD, fg=fg, padx=10, pady=6,
                       anchor="w", cursor="hand2")
        btn.bind("<Button-1>", lambda e: cmd())
        btn.bind("<Enter>",    lambda e: btn.config(bg=BORDER))
        btn.bind("<Leave>",    lambda e: btn.config(bg=CARD))
        return btn

    # ── Game logic ─────────────────────────────────────────────────
    def _new_round(self):
        """Pick a random category and start revealing its drawing."""
        if self._game_over:
            return

        # Cancel any pending reveal
        if self._reveal_job:
            self.root.after_cancel(self._reveal_job)
            self._reveal_job = None

        # Start timer on very first round (timed mode)
        if self.timed_mode and self.round_num == 0:
            self._start_timer()

        self.secret_label  = random.choice(self.categories)
        self.wrong_guesses = 0
        self.round_num    += 1
        self.round_var.set(str(self.round_num))
        self.guess_history = []
        self._refresh_hist()

        self.result_emoji.config(text="❓")
        self.result_lbl.config(text="?", fg=ACCENT)
        self.result_sub.config(text="")
        self.status_lbl.config(text="watch carefully…", fg=TEXT_LO)
        self.prompt_lbl.config(text="What is the AI drawing?", fg=TEXT_MID)
        self.guess_var_entry.set("")
        self.entry.config(state="normal")
        self.entry.focus_set()

        # Generate pixel image from a random centroid of the chosen class
        self._generate_pixel_image()
        self._start_reveal()

    def _generate_pixel_image(self):
        """
        Pick a random real sample from pixel_prototypes for the secret label.
        These are genuine 28x28 QuickDraw images in [0,1] float range —
        no PCA reconstruction needed, so they look clean and recognisable.
        """
        samples = self.pixel_prototypes.get(self.secret_label, [])
        if samples:
            self.secret_pixels = samples[random.randint(0, len(samples) - 1)].copy()
        else:
            self.secret_pixels = np.zeros((28, 28), dtype=np.float32)

    def _start_reveal(self):
        """
        Upscale the 28x28 image to canvas size using Lanczos, then reveal
        it progressively by drawing vertical slices of the smooth image.
        """
        self.canvas.delete("all")
        self._revealed_cols = 0

        # Upscale 28x28 → canvas_size using Lanczos for smooth result
        pil_img = Image.fromarray(
            (self.secret_pixels * 255).astype(np.uint8), mode="L"
        ).resize((self.canvas_size, self.canvas_size), Image.LANCZOS)

        # Convert to RGBA with dark background replacing black
        pil_rgba = pil_img.convert("RGB")
        self._reveal_photo_data = np.array(pil_rgba)  # (H, W, 3) uint8

        # Build a fully-dark starting canvas image
        dark = Image.new("RGB", (self.canvas_size, self.canvas_size), BG_RGB)
        self._reveal_display = dark.copy()
        self._reveal_pil = pil_rgba

        self._reveal_step()

    def _reveal_step(self):
        """Reveal one more vertical slice of the smooth upscaled image."""
        total_cols = self.canvas_size
        slice_w = max(1, total_cols // self.REVEAL_STEPS)

        if self._revealed_cols >= total_cols:
            self.status_lbl.config(text="type your guess below!", fg=WARNING)
            return

        # Paste the next slice from the real image onto the display image
        x0 = self._revealed_cols
        x1 = min(total_cols, x0 + slice_w)
        region = self._reveal_pil.crop((x0, 0, x1, self.canvas_size))
        self._reveal_display.paste(region, (x0, 0))

        # Re-render full display image as a PhotoImage
        self._tk_img = ImageTk.PhotoImage(self._reveal_display)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        self._revealed_cols = x1
        delay = max(10, self.REVEAL_DELAY - self._revealed_cols // 10)
        self._reveal_job = self.root.after(delay, self._reveal_step)

    def _check_guess(self):
        guess = self.guess_var_entry.get().strip().lower()
        if not guess:
            return

        correct = (guess == self.secret_label.lower())

        self.guess_history.insert(0, ("✓ " + guess if correct else "✗ " + guess, correct))
        self.guess_history = self.guess_history[:8]
        self._refresh_hist()
        self.guess_var_entry.set("")

        if correct:
            self._on_correct()
        else:
            self.wrong_guesses += 1
            self.status_lbl.config(
                text=f"nope! ({self.wrong_guesses} wrong)", fg=DANGER)
            self.result_emoji.config(text="❌")
            self.result_lbl.config(text=guess.upper(), fg=DANGER)

    def _on_correct(self):
        if self._game_over:
            return
        self.score += 1
        self.score_var.set(str(self.score))

        emoji = EMOJI_MAP.get(self.secret_label, "🎨")
        self.result_emoji.config(text=emoji)
        self.result_lbl.config(text="CORRECT!", fg=SUCCESS)
        self.result_sub.config(text=f"It was: {self.secret_label}")
        self.status_lbl.config(text="🎉  correct! next round in 1 s…", fg=SUCCESS)
        self.prompt_lbl.config(
            text=f"✓ It was a {self.secret_label}!", fg=SUCCESS)
        self.entry.config(state="disabled")
        # Timed mode: faster turn-around so seconds aren't wasted
        delay = 800 if self.timed_mode else 2000
        self.root.after(delay, self._new_round)

    # ── Timer (timed mode) ─────────────────────────────────────────
    def _start_timer(self):
        self._time_left = TIMER_SECONDS
        self._tick()

    def _tick(self):
        if self._game_over:
            return
        self._time_left -= 1
        if hasattr(self, "timer_var"):
            self.timer_var.set(str(max(0, self._time_left)))
            # Colour shifts: green → yellow → red as time runs out
            if self._time_left > 15:
                self.timer_lbl.config(fg=SUCCESS)
            elif self._time_left > 7:
                self.timer_lbl.config(fg=WARNING)
            else:
                self.timer_lbl.config(fg=DANGER)

        if self._time_left <= 0:
            self._end_game()
        else:
            self._timer_job = self.root.after(1000, self._tick)

    def _end_game(self):
        """Called when timer hits 0.  Show game-over overlay and save score."""
        self._game_over = True

        # Cancel any running reveal
        if self._reveal_job:
            self.root.after_cancel(self._reveal_job)
            self._reveal_job = None

        # Disable input
        try:
            self.entry.config(state="disabled")
        except Exception:
            pass

        # ── Save to leaderboard ────────────────────────────────
        import datetime
        entry = {
            "score": self.score,
            "date":  datetime.datetime.now().strftime("%b %d  %H:%M"),
        }
        AI_DRAWS_LEADERBOARD.append(entry)
        AI_DRAWS_LEADERBOARD.sort(key=lambda x: x["score"], reverse=True)
        del AI_DRAWS_LEADERBOARD[3:]   # keep top 3

        # ── Build overlay ──────────────────────────────────────
        overlay = tk.Frame(self.root, bg="#000000")
        overlay.place(x=0, y=0, relwidth=1, relheight=1)
        self._overlay = overlay

        f_big  = tkfont.Font(family="Georgia",     size=48, weight="bold")
        f_med  = tkfont.Font(family="Georgia",     size=22, weight="bold")
        f_sm   = tkfont.Font(family="Courier New", size=11)
        f_lb   = tkfont.Font(family="Courier New", size=13)
        f_btn  = tkfont.Font(family="Courier New", size=12, weight="bold")

        inner = tk.Frame(overlay, bg=SURFACE,
                         highlightthickness=2,
                         highlightbackground=WARNING)
        inner.place(relx=0.5, rely=0.5, anchor="center")

        # ── Header ─────────────────────────────────────────────
        tk.Label(inner, text="⏱  TIME'S UP!", font=f_big,
                 bg=SURFACE, fg=WARNING).pack(pady=(28, 4))
        tk.Label(inner,
                 text=f"You guessed  {self.score}  correct  in  {TIMER_SECONDS}s",
                 font=f_med, bg=SURFACE, fg=TEXT_HI).pack(pady=(0, 20))

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=30, pady=4)

        # ── Leaderboard ────────────────────────────────────────
        tk.Label(inner, text="🏆  TOP SCORES", font=f_sm,
                 bg=SURFACE, fg=WARNING).pack(pady=(12, 4))

        medals = ["🥇", "🥈", "🥉"]
        colors_lb = [WARNING, TEXT_MID, ACCENT3]
        for i, lb_entry in enumerate(AI_DRAWS_LEADERBOARD[:3]):
            medal = medals[i] if i < len(medals) else "  "
            fg_c  = colors_lb[i] if i < len(colors_lb) else TEXT_LO
            # Highlight the current score row
            is_new = (lb_entry is entry)
            row_bg = CARD if not is_new else "#1e2a10"
            row_frame = tk.Frame(inner, bg=row_bg, padx=24, pady=8,
                                 highlightthickness=1 if is_new else 0,
                                 highlightbackground=SUCCESS)
            row_frame.pack(fill="x", padx=40, pady=3)
            tag = "  ← YOU" if is_new else ""
            tk.Label(row_frame,
                     text=f"{medal}  {lb_entry['score']} correct   —   {lb_entry['date']}{tag}",
                     font=f_lb, bg=row_bg, fg=fg_c).pack()

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=30, pady=12)

        # ── Buttons ────────────────────────────────────────────
        btn_row = tk.Frame(inner, bg=SURFACE)
        btn_row.pack(pady=(0, 28))

        def _play_again():
            overlay.destroy()
            self._game_over    = False
            self._time_left    = TIMER_SECONDS
            self.score         = 0
            self.round_num     = 0
            self.score_var.set("0")
            self.round_var.set("0")
            if hasattr(self, "timer_var"):
                self.timer_var.set(str(TIMER_SECONDS))
                self.timer_lbl.config(fg=SUCCESS)
            self._new_round()
            self._start_timer()

        play_btn = tk.Label(btn_row, text="  ▶  PLAY AGAIN  ",
                            font=f_btn, bg=SUCCESS, fg=BG,
                            padx=18, pady=10, cursor="hand2")
        play_btn.pack(side="left", padx=10)
        play_btn.bind("<Button-1>", lambda e: _play_again())
        play_btn.bind("<Enter>",    lambda e: play_btn.config(bg=ACCENT3))
        play_btn.bind("<Leave>",    lambda e: play_btn.config(bg=SUCCESS))

        if self.on_switch_mode:
            menu_btn = tk.Label(btn_row, text="  ⇄  MAIN MENU  ",
                                font=f_btn, bg=CARD, fg=TEXT_MID,
                                padx=18, pady=10, cursor="hand2")
            menu_btn.pack(side="left", padx=10)
            menu_btn.bind("<Button-1>", lambda e: self.on_switch_mode())
            menu_btn.bind("<Enter>",    lambda e: menu_btn.config(bg=BORDER))
            menu_btn.bind("<Leave>",    lambda e: menu_btn.config(bg=CARD))

    def _skip_round(self):
        if self._game_over:
            return
        emoji = EMOJI_MAP.get(self.secret_label, "🎨")
        self.result_emoji.config(text=emoji)
        self.result_lbl.config(text=self.secret_label.upper(), fg=WARNING)
        self.result_sub.config(text="skipped — no point awarded")
        self.status_lbl.config(text=f"Answer: {self.secret_label}. Next round in 2 s…",
                               fg=WARNING)
        self.entry.config(state="disabled")
        if self._reveal_job:
            self.root.after_cancel(self._reveal_job)
        self.root.after(2000, self._new_round)

    def _show_hint(self):
        label = self.secret_label
        hint  = f"It starts with '{label[0].upper()}'"
        if self.wrong_guesses >= self.HINT_THRESHOLD:
            hint += f"  ({len(label)} letters)"
        self.status_lbl.config(text=f"💡 Hint: {hint}", fg=ACCENT3)

    def _refresh_hist(self):
        for i, lbl_widget in enumerate(self.hist_labels):
            if i < len(self.guess_history):
                text, correct = self.guess_history[i]
                color = SUCCESS if correct else DANGER
                lbl_widget.config(text=f"  {text}", fg=color)
            else:
                lbl_widget.config(text="")

    # ── Background animation ───────────────────────────────────────
    def _animate(self):
        if not self.anim_running:
            return
        self.glow_phase = (self.glow_phase + 0.03) % (2 * math.pi)

        r = int(200 + 55 * math.sin(self.glow_phase))
        g = int(80  + 30 * math.sin(self.glow_phase + 2))
        b = int(100 + 40 * math.sin(self.glow_phase + 4))
        glow_color = f"#{r:02x}{g:02x}{b:02x}"
        self.glow_frame.config(bg=glow_color)

        self.bg_canvas.delete("particle")
        for p in self.particles:
            p.update()
            alpha_hex = format(int(p.a * 60), '02x')
            col = f"#3{alpha_hex}3{alpha_hex}ff"
            x, y, r2 = p.x, p.y, p.r
            try:
                self.bg_canvas.create_oval(x-r2, y-r2, x+r2, y+r2,
                                           fill=col, outline="", tags="particle")
            except Exception:
                pass
        self.root.after(40, self._animate)

    def _on_close(self):
        self.anim_running = False
        self.root.destroy()