"""Auto-player for the "Pick the Lock" minigame (Dota 2, Dark Carnival).

Works purely from the screen: the lock dial is captured with DXGI desktop
duplication, the pick and the yellow/blue bars are located in a polar unwrap
of the ring, and a left click is sent when the pick is predicted to be on a
bar at the moment the game registers the click. Nothing touches the game
process; the only input is synthetic mouse events.

Hotkeys (global):
    F5   calibrate - mouse over the centre of the dial
    F6   calibrate - mouse over the outer edge of the dark ring
    F7   start / stop
    F8   debug view + recording (recordings/)
    DEL  quit
"""
import base64
import ctypes
import ctypes.wintypes as wt
import json
import math
import os
import queue
import sys
import threading
import time
import winsound
import zlib
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# Angles are in degrees, 0 = 3 o'clock, increasing clockwise (screen y points down).
# OpenCV hues are 0-180.
DEFAULTS = {
    "center": None,                # dial centre, physical screen pixels
    "radius": None,                # outer radius of the dark ring, pixels
    "capture": "auto",             # auto | dxcam | mss
    "fps_profile": "auto",         # auto | 120 | 30, see FPS_PROFILES

    # vision
    "band_inner": 0.62,            # analysed ring band, fraction of radius
    "band_outer": 0.90,
    "angle_bins": 720,
    "yellow_hue": [14, 30],        # yellow bars sit at 18-24
    "blue_hue": [98, 112],         # blue bars 104-109; the pick's glow is 117-125
    "bar_delta_s_min": 90,         # bars are detected on the light they add to the
    "bar_delta_v_min": 20,         # ring, which keeps fading bars visible
    "bar_fill_frac": 0.45,
    "bar_min_width_deg": 1.0,
    "bar_merge_gap_deg": 2.5,      # fading bars break into pieces
    "pick_mask_deg": 10.0,         # the pick's glow is never a bar
    "pick_min_score": 12.0,
    "bg_tau_ms": 30,
    "ring_tau_ms": 120,
    "effect_delta": 10,            # ring colour shift that means a hit/miss flash
    "effect_hold_ms": 150,
    "effect_reset_ms": 1200,       # a shift lasting this long is lighting, not a flash
    "read_timer": True,

    # tracking
    "max_speed_dps": 720,
    "settle_max_resid_deg": 0.6,   # a track must move in a straight line to be trusted
    "glitch_settle_deg": 6.0,      # ... and this far after a turnaround we didn't cause

    # clicking
    "auto_latency": True,
    "edge_margin_deg": 0.6,
    "min_target_deg": 2.5,
    "min_target_s": 0.025,         # skip bars the pick crosses faster than this
    "min_click_interval_ms": 200,
    "hit_confirm_ms": 250,
    "miss_lockout_ms": 600,        # the game disables the pick after a miss anyway
    "click_hold_ms": 12,
    "click_blue": True,
    "boost_mode": "always",        # always | off  (right mouse button held)
    "move_cursor_on_start": True,
    "require_dota_foreground": True,

    # strategy
    "strategy": "plan",            # plan | greedy
    "yellow_points": 1000,
    "blue_seconds": 1.5,           # a blue bar scores like a yellow one and adds time,
    "points_per_second": 1600,     # valued at the late-game scoring rate
    "bar_shrink_dps": 6.2,
    "plan_horizon_s": 2.5,
    "plan_depth": 6,
}

# How often Dota redraws the minigame (fps_max_ui) is how often we see it.
# Durations are in ms; the code enforces a minimum number of frames on top.
FPS_PROFILES = {
    "120": {
        "expected_fps": 120,
        "bar_persist_ms": 25,
        "settle_ms": 30,
        "track_lost_ms": 40,
        "vel_window_ms": 50,
        "latency_default_ms": 25,
        "latency_learn_max_frame_ms": 15,
    },
    "30": {
        "expected_fps": 30,
        "bar_persist_ms": 60,
        "settle_ms": 90,
        "track_lost_ms": 110,
        "vel_window_ms": 130,
        "latency_default_ms": 42,      # clicks also wait for the next game frame
        "latency_learn_max_frame_ms": 45,
    },
}


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


CONFIG_PATH = os.path.join(app_dir(), "config.json")


def latency_key(profile):
    return "latency_ms" if profile == "120" else f"latency_ms_{profile}"


# Calibration and learned latency are saved along with the user's own settings.
# Defaults are never written, so a newer build's defaults always apply.
RUNTIME_KEYS = ("center", "radius") + tuple(latency_key(p) for p in FPS_PROFILES)


def apply_profile(cfg, name, user_keys=()):
    for key, value in FPS_PROFILES[name].items():
        if key not in user_keys:
            cfg[key] = value
    cfg["active_profile"] = name


def make_config(profile="120", **overrides):
    cfg = dict(DEFAULTS)
    apply_profile(cfg, profile)
    cfg.update(overrides)
    return cfg


def load_config():
    user = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                user = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[!] ignoring unreadable config.json: {e}")
    cfg = dict(DEFAULTS)
    cfg.update(user)
    if cfg["fps_profile"] not in FPS_PROFILES:
        cfg["fps_profile"] = "auto"
    apply_profile(cfg, "120" if cfg["fps_profile"] == "auto" else cfg["fps_profile"], user)
    return cfg, set(user)


def save_config(cfg, user_keys):
    out = {k: v for k, v in cfg.items() if (k in user_keys or k in RUNTIME_KEYS) and v is not None}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def circ_smooth(x, k):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.concatenate([x[-pad:], x, x[:pad]])
    return np.convolve(xp, np.ones(k, np.float32) / k, mode="valid")


def circular_runs(mask):
    """(start, length) of each run of True in a circular array."""
    n = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [(0, n)]
    k = int(np.argmin(mask))
    edges = np.diff(np.concatenate([[0], np.roll(mask, -k).astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [((s + k) % n, e - s) for s, e in zip(starts, ends)]


def merge_runs(runs, gap, n):
    if len(runs) < 2:
        return runs
    runs = sorted(runs)
    out = [list(runs[0])]
    for s, length in runs[1:]:
        if s - (out[-1][0] + out[-1][1]) <= gap:
            out[-1][1] = s + length - out[-1][0]
        else:
            out.append([s, length])
    if len(out) > 1 and (out[0][0] + n) - (out[-1][0] + out[-1][1]) <= gap:
        last = out.pop()
        out[0] = [last[0], out[0][0] + n + out[0][1] - last[0]]
    return [tuple(r) for r in out]


# Glyphs 0-9 of the TIME REMAINING display, 16x24, zlib + base64.
_DIGITS = (
    "eNq9l39IXVUcwI/bW76Z0uOlZu3ZM7osWmsi1JxUsgLn3kKTZJB/zGgmOOiOtGKNsixqkS6Da8sykRQCG2y00e40yLAf47XV"
    "qC0JvEMXq1xvuGeaujQ7fb/n3HPOPbdtuYi+8O49n3PO/X6/93vP+X7PIwQk1kVpV4y4ktcxRUGmOjimH6GuPMJ4s0A6VQCY"
    "M4rNri68juYQUgX3ZEkg8CZ2VJGADTcb5oVQzV4Swe7NQo8TKhN6SAFajbTC5Qw3fAaajcgnFLci13PuQaaK6xfBx/3jrYvn"
    "TwUfVP6URXR/I5ljlJ7LYUGdhxcPkA7orUWuhQaEtAhuY3kQ5TEeLynrty/AkAvraocnWIyhHblly+GkiHlKkfUz9cgFqkvy"
    "YvzrUO2sy+Pzc7OPFRMieMWDKcygYOHNf831Pr7bx3cs6Jz/x//Cb5Ct5gnWmjbXY78zL0M3sS9MiBbMvRk6014fTxTQgTYD"
    "pOwz3vHO20FuL+Mj3T4p93HYx0v36Jxx6vLzub4GyWxDng0JjOLqpN/K4WcRf1gpcAN7ulVg5Bhi23JhK85e7kYXl/GMUCJm"
    "lyDNbpC6cOVOVgm/N57F6D4qRnvZs3XSkeoX+YsMmfks/nMymInONF/829N88e4ETlog33B2iF0URTVZNdPIc9XSTgObYEqO"
    "LYanywWGMI/RIwIrB5nPzJYlvLGyiIeP1bhzBdvGMqLNp09zXZWrLLfvFWnesH3+Z7GOIZUvDP37/43LdU5jHickd/relxjJ"
    "HuIVW+Q1LkVJjUMO1diTt1kNSWrMa4hiXkOkPV5DRJ4lmDNHPWyM0f2ePEwgiZfpeXonUZzj0ONBxcH9dH4bIfcILuJ2Iy6n"
    "g7J7PdwI9xgIbh8nttMZpVcmhitobxfc5Y7yvp+XHe/Tzr/j0xaTQcG2+42MKAmdVoxi4mM1PjaukLvR2irFrkBZIe3e2nMg"
    "y7dfDxCTy6A7o1joLU761mM11ddr1jHf+rYvz3z+kE+fWanZS+SLYZaFXpKzK7F+HRT5j9yGOWz4eoHBk4B98llyCPBkUOLz"
    "fwLfKbEYdR1KI57VTUekLnIfWq5Q7/Ee4MDVEqMJShe2GHvs2/MYb4fh37vZ8espwOVfesL7JCHZ3nBPl5JtrDHnOL/g/cdQ"
    "M8vVkFZWf4ithxhvQsUBzKHNyN38SPA4NM+PwOUF7kmuUJvrYZz/gI8txSOFuDyzEVOacJj5O4C8Cadv/IftdMrHDT4O6thC"
    "xP4z7voJ+CYV1oexgmUoRj/aFLICeJ3iDwDfXypx9Tjwq2q4GKdHFfd70xkha7D6rNOz45Ry5oYRdhZW01HbGsVvAX6cqnLF"
    "GDtK63twrToZHAXsV8NRqr4Jyg5cnWE1/TvgZ7RUb4JGU249y00xjuXf4aask67E3AIdt6w4335stgnWwqykGU1yU7L61NSk"
    "0tEl2c032AobGHgr7PoxTgibEI/xKloI/XHNHTAsHeZqwrZgGzticd1/HI3LqsysOCxYBobBdFT6Y0PeHEN9Z9hLsen5bLb0"
    "kwWSvXp+Hzb7tvJpmY3n+GPzn2QCpvYod3tS+a5U24GkfYX3/paWftaRfT9ea9OhsOH/Dfrcu3DZtwT1XoVqEelubmj3xfaz"
    "nyfht0OtI/o67oslMo/Rl/FSGoD0UcoSTW4C/2+0VVTUsbdoJ7uYyslJrrqZXPM51UtG9Ht29L8wA5evMUtmv3Z0Zma47jDw"
    "E+4GKCxcueI3+Px5errvVbj2PJwvctWBA6wstEtMqQD7X1wr+Vb4D5W4GVt/Ae3agts="
)


class TimerReader:
    """Reads the TIME REMAINING display by template matching.

    The decimal point is too faint to segment reliably, but the display always
    shows one decimal, so the value is the digits read as an integer / 10.
    """

    GW, GH = 16, 24
    MAX_DIST = 9000

    def __init__(self, radius):
        raw = zlib.decompress(base64.b64decode(_DIGITS))
        self.templates = np.frombuffer(raw, np.uint8).reshape(10, self.GH, self.GW).astype(np.float32)
        k = radius / 225.18            # box measured on a dial of radius 225
        self.box = (int(-24 * k), int(38 * k), int(-84 * k), int(86 * k))

    def read(self, roi):
        cy, cx = roi.shape[0] // 2, roi.shape[1] // 2
        y0, y1, x0, x1 = self.box
        crop = roi[cy + y0:cy + y1, cx + x0:cx + x1]
        _, binary = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 130, 255, cv2.THRESH_BINARY)
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        height = crop.shape[0]
        glyphs = sorted((stats[i][0], i) for i in range(1, n)
                        if stats[i][3] > 0.45 * height and stats[i][4] >= 8)
        if not 1 <= len(glyphs) <= 3:
            return None
        digits = ""
        for _, i in glyphs:
            x, y, w, h, _ = stats[i]
            glyph = cv2.resize(binary[y:y + h, x:x + w], (self.GW, self.GH), interpolation=cv2.INTER_AREA)
            dist = np.abs(self.templates - glyph.astype(np.float32)).sum(axis=(1, 2))
            best = int(np.argmin(dist))
            if dist[best] > self.MAX_DIST:
                return None
            digits += str(best)
        return int(digits) / 10.0


@dataclass(slots=True)
class Bar:
    start: float
    width: float
    kind: str                       # "yellow" | "blue"

    @property
    def center(self):
        return (self.start + self.width / 2.0) % 360.0


@dataclass(slots=True)
class Detection:
    t: float
    pick: float | None
    score: float
    bars: list = field(default_factory=list)
    effect: bool = False            # a hit/miss flash is on screen
    flash: str | None = None        # "hit" (blue-white) | "miss" (red)


class Detector:
    """Finds the pick and the bars in a square crop centred on the dial.

    The ring is unwrapped to polar coordinates (rows = angle). The pick is the
    narrow moving peak against a background model that is frozen under the
    pick; bars are the yellow/blue light added on top of the ring's median
    colour, with the pick's own pixels taken from the background.
    """

    NR = 96

    def __init__(self, cfg, radius):
        self.cfg = cfg
        self.R = float(radius)
        self.na = int(cfg["angle_bins"])
        self.deg_per_bin = 360.0 / self.na
        self.c0 = int(cfg["band_inner"] * self.NR)
        self.c1 = max(self.c0 + 2, int(cfg["band_outer"] * self.NR))
        self.k_small = 3
        self.k_large = max(9, int(round(20.0 / self.deg_per_bin)) | 1)
        s_min, v_min = cfg["bar_delta_s_min"], cfg["bar_delta_v_min"]
        self.yellow_lo = np.array([cfg["yellow_hue"][0], s_min, v_min], np.uint8)
        self.yellow_hi = np.array([cfg["yellow_hue"][1], 255, 255], np.uint8)
        self.blue_lo = np.array([cfg["blue_hue"][0], s_min, v_min], np.uint8)
        self.blue_hi = np.array([cfg["blue_hue"][1], 255, 255], np.uint8)
        self.frame_dt = 1.0 / cfg["expected_fps"]
        self.last_t = None
        self.reset()

    def reset(self):
        self.bg = None
        self.seen_frames = None
        self.ring_ref = None
        self.effect_since = None

    def _alpha(self, tau_ms):
        return 1.0 - math.exp(-self.frame_dt / (tau_ms / 1000.0))

    def frames_for(self, ms, minimum):
        return max(minimum, int(round(ms / 1000.0 / self.frame_dt)))

    def _rows_around(self, angle, half_deg):
        i = int(round(angle / self.deg_per_bin))
        h = int(math.ceil(half_deg / self.deg_per_bin))
        return np.arange(i - h, i + h + 1) % self.na

    def process(self, roi, t, predicted=None, search_deg=None, moving=False):
        if self.last_t is not None and 0 < t - self.last_t < 0.2:
            self.frame_dt += 0.1 * ((t - self.last_t) - self.frame_dt)
        self.last_t = t

        h, w = roi.shape[:2]
        polar = cv2.warpPolar(roi, (self.NR, self.na), (w / 2.0, h / 2.0), self.R,
                              cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR)
        band = np.ascontiguousarray(polar[:, self.c0:self.c1])
        bandf = band.astype(np.float32)
        if self.bg is None or self.bg.shape != bandf.shape:
            self.bg = bandf.copy()

        pick, score = self._find_pick(bandf, predicted, search_deg)

        # Bar colour is judged with the pick's pixels replaced by background.
        comp = band
        update_mask = np.full(band.shape[:2], 255, np.uint8)
        if pick is not None:
            rows = self._rows_around(pick, self.cfg["pick_mask_deg"])
            comp = band.copy()
            comp[rows] = np.clip(self.bg[rows], 0, 255).astype(np.uint8)
            if moving:                 # freezing under a still "pick" would lock in a ghost
                update_mask[rows] = 0
        base = np.median(comp, axis=0)
        added = np.clip(comp.astype(np.int16) - base.astype(np.int16), 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(added, cv2.COLOR_BGR2HSV)
        frac_yellow = cv2.inRange(hsv, self.yellow_lo, self.yellow_hi).mean(axis=1) / 255.0
        frac_blue = cv2.inRange(hsv, self.blue_lo, self.blue_hi).mean(axis=1) / 255.0

        effect, flash = self._check_flash(np.median(base, axis=0), t)
        bars = self._find_bars(frac_yellow, frac_blue)

        cv2.accumulateWeighted(bandf, self.bg, self._alpha(self.cfg["bg_tau_ms"]), update_mask)
        return Detection(t, pick, score, bars, effect, flash)

    def _find_pick(self, bandf, predicted, search_deg):
        profile = np.abs(bandf - self.bg).max(axis=2).mean(axis=1)
        dog = circ_smooth(profile, self.k_small) - circ_smooth(profile, self.k_large)
        threshold = self.cfg["pick_min_score"]
        idx = None
        if predicted is not None:
            rows = self._rows_around(predicted, search_deg or 15.0)
            j = rows[int(np.argmax(dog[rows]))]
            if dog[j] >= threshold:
                idx = j
        if idx is None:
            j = int(np.argmax(dog))
            if dog[j] >= threshold * (1.0 if predicted is None else 1.5):
                idx = j
        if idx is None:
            return None, 0.0
        a, b, c = dog[(idx - 1) % self.na], dog[idx], dog[(idx + 1) % self.na]
        den = a - 2 * b + c
        offset = 0.5 * (a - c) / den if den < -1e-6 else 0.0
        return ((idx + max(-0.5, min(0.5, offset))) * self.deg_per_bin) % 360.0, float(b)

    def _check_flash(self, ring, t):
        if self.ring_ref is None:
            self.ring_ref = ring.copy()
        effect = bool(np.abs(ring - self.ring_ref).max() > self.cfg["effect_delta"])
        flash = None
        if effect:
            # A miss tints the ring red. A fading hit flash turns warm-white,
            # but unlike the miss its green channel goes up as well.
            red_shift = (ring[2] - ring[1]) - (self.ring_ref[2] - self.ring_ref[1])
            green_up = ring[1] - self.ring_ref[1]
            flash = "miss" if red_shift > 8 and green_up <= 3 else "hit"
        if not effect:
            self.effect_since = None
            self.ring_ref += self._alpha(self.cfg["ring_tau_ms"]) * (ring - self.ring_ref)
        else:
            if self.effect_since is None:
                self.effect_since = t
            if t - self.effect_since > self.cfg["effect_reset_ms"] / 1000.0:
                self.ring_ref = ring.copy()
        return effect, flash

    def _find_bars(self, frac_yellow, frac_blue):
        ff = self.cfg["bar_fill_frac"]
        raw = np.stack([frac_yellow >= ff, (frac_blue >= ff) & (frac_yellow < ff)])
        if self.seen_frames is None or self.seen_frames.shape != raw.shape:
            self.seen_frames = np.zeros(raw.shape, np.int16)
        self.seen_frames = np.where(raw, self.seen_frames + 1, 0)
        stable = self.seen_frames >= self.frames_for(self.cfg["bar_persist_ms"], 2)
        gap = int(round(self.cfg["bar_merge_gap_deg"] / self.deg_per_bin))
        bars = []
        for kind, mask in (("yellow", stable[0]), ("blue", stable[1])):
            for s, n in merge_runs(circular_runs(mask), gap, self.na):
                if n * self.deg_per_bin >= self.cfg["bar_min_width_deg"]:
                    bars.append(Bar(s * self.deg_per_bin, n * self.deg_per_bin, kind))
        return bars


class Controller:
    """Tracks the pick and decides when to click and whether to hold boost.

    Time is passed in rather than read from a clock, so the same code runs
    against live capture, recordings and the simulators.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.lat_samples = deque(maxlen=9)
        self.speed_samples = deque(maxlen=240)
        self.hits = self.misses = self.clicks = self.skipped = 0
        self.time_left = None          # from the timer OCR, None if unknown
        self.load_latency()
        self.reset()

    def load_latency(self):
        c = self.cfg
        self.latency = c.get(latency_key(c["active_profile"]), c["latency_default_ms"]) / 1000.0
        self.lat_samples.clear()

    def reset(self):
        self.hist = deque(maxlen=32)   # (t, unwrapped angle)
        self.last_angle = None
        self.unwrapped = 0.0
        self.direction = 0
        self.omega = 0.0
        self.frame_dt = 1.0 / self.cfg["expected_fps"]
        self.last_t = None
        self.last_seen_t = -1e9
        self.last_good_t = -1e9
        self.last_click_t = -1e9
        self.lockout_until = -1e9
        self.effect_until = -1e9
        self.stale = False
        self.glitched = False          # track turned around without a click of ours
        self.pending = None            # click awaiting its hit/miss flash
        self.target = None
        self.boost_on = False

    # tracking

    def predicted_angle(self, t):
        if self.last_angle is None or len(self.hist) < 2:
            return self.last_angle
        return (self.last_angle + self.omega * (t - self.hist[-1][0])) % 360.0

    def moving(self):
        return self.last_angle is not None and len(self.hist) >= 3 and abs(self.omega) > 20

    def search_window(self):
        return max(12.0, abs(self.omega) * self.frame_dt * 2.5)

    def frames_for(self, ms, minimum):
        return max(minimum, int(round(ms / 1000.0 / self.frame_dt)))

    def _fit_omega(self):
        if len(self.hist) < 2:
            return 0.0
        window = self.cfg["vel_window_ms"] / 1000.0
        pts = [p for p in self.hist if self.hist[-1][0] - p[0] <= window]
        if len(pts) < 3:
            pts = list(self.hist)[-3:]
        ts = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        ts = ts - ts.mean()
        den = (ts * ts).sum()
        return float((ts * (ys - ys.mean())).sum() / den) if den > 0 else 0.0

    def _forget_track(self):
        self.hist.clear()
        self.last_angle = None
        self.direction = 0
        self.glitched = False

    def _track(self, det):
        """Returns (tracked, reversed)."""
        t = det.t
        if self.last_t is not None and 0 < t - self.last_t < 0.1:
            self.frame_dt = 0.9 * self.frame_dt + 0.1 * (t - self.last_t)
        self.last_t = t

        if det.effect:
            self.effect_until = t + self.cfg["effect_hold_ms"] / 1000.0
        if t < self.effect_until:
            self.stale = True
            return False, False
        if self.stale:                 # the pick has usually turned around meanwhile
            self.stale = False
            self._forget_track()

        if det.pick is not None and self.last_angle is not None and len(self.hist) >= 3:
            gap = t - self.hist[-1][0]
            expected = (self.last_angle + self.omega * gap) % 360.0
            if abs(wrap180(det.pick - expected)) > max(4.0, self.cfg["max_speed_dps"] * gap):
                det.pick = None
        if det.pick is None:
            if t - self.last_seen_t > max(self.cfg["track_lost_ms"] / 1000.0, 2.5 * self.frame_dt):
                self._forget_track()
                self.omega = 0.0
            return False, False
        self.last_seen_t = t

        reversed_ = False
        if self.last_angle is None:
            self.unwrapped = det.pick
        else:
            step = wrap180(det.pick - self.last_angle)
            if self.direction != 0 and step * self.direction < -0.4 and abs(self.omega) > 25:
                reversed_ = True
                # The pick only turns around when a click registers, so a
                # turnaround long after our last click is a tracking glitch,
                # typically sparkles from the previous hit.
                self.glitched = t - self.last_click_t > 0.15
                turn = self.hist[-1]
                self.hist.clear()
                self.hist.append(turn)
            if abs(step) > 0.4:
                self.direction = 1 if step > 0 else -1
            self.unwrapped += step
        self.last_angle = det.pick
        self.hist.append((t, self.unwrapped))
        self.omega = self._fit_omega()
        return True, reversed_

    def _steady(self):
        """True once the track is trustworthy: enough samples since the last
        turnaround, no speed change in progress, straight-line motion, and a
        few degrees of travel after a turnaround we didn't cause."""
        c = self.cfg
        if len(self.hist) < self.frames_for(c["settle_ms"], 3):
            return False
        (t0, a0), (t1, a1) = self.hist[-3], self.hist[-1]
        if t1 <= t0:
            return False
        if abs((a1 - a0) / (t1 - t0) - self.omega) > 0.2 * abs(self.omega) + 10:
            return False
        if self.glitched and abs(self.hist[-1][1] - self.hist[0][1]) < c["glitch_settle_deg"]:
            return False
        pts = list(self.hist)[-8:]
        ts = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        resid = ys - np.polyval(np.polyfit(ts, ys, 1), ts)
        return float(np.sqrt(np.mean(resid ** 2))) <= c["settle_max_resid_deg"]

    def _speed_plausible(self, speed):
        # With boost held the pick's speed barely varies; anything else is
        # most likely debris from the last hit.
        if len(self.speed_samples) < 60:
            return True
        typical = float(np.median(self.speed_samples))
        lo, hi = (0.75, 1.3) if self.cfg["boost_mode"] == "always" else (0.3, 3.0)
        return lo * typical <= speed <= hi * typical

    # strategy

    def dead_time(self):
        settle = max(self.cfg["settle_ms"] / 1000.0, 3 * self.frame_dt)
        return 0.03 + self.cfg["effect_hold_ms"] / 1000.0 + settle

    def min_width(self, speed):
        # The game only registers clicks on its own frames, hence half a frame.
        c = self.cfg
        return max(c["min_target_deg"], speed * (c["min_target_s"] + 0.5 * self.frame_dt))

    def horizon(self):
        h = self.cfg["plan_horizon_s"]
        return h if self.time_left is None else max(0.05, min(h, self.time_left))

    def _bar_value(self, kind):
        c = self.cfg
        if kind == "yellow":
            return c["yellow_points"]
        return c["yellow_points"] + c["blue_seconds"] * c["points_per_second"]

    def _plan(self, theta, direction, t0, bars, speed, min_w, depth):
        """Best value reachable within the horizon. At each bar the pick meets
        we either hit it (collect, reverse, dead time) or let it pass.
        `bars` holds (centre, width now, kind); widths shrink at a known rate."""
        if depth == 0 or not bars:
            return 0.0
        nearest = None
        for i, (center, width, _) in enumerate(bars):
            dist = ((center - theta) * direction) % 360.0
            arrive = t0 + dist / speed
            if arrive > self.horizon() or width - self.cfg["bar_shrink_dps"] * arrive < min_w:
                continue
            if nearest is None or dist < nearest[0]:
                nearest = (dist, arrive, i)
        if nearest is None:
            return 0.0
        _, arrive, i = nearest
        center, _, kind = bars[i]
        dead = self.dead_time()
        hit = (self._bar_value(kind) * (1.0 - 0.02 * arrive)
               + self._plan((center - direction * speed * dead) % 360.0, -direction, arrive + dead,
                            bars[:i] + bars[i + 1:], speed, min_w, depth - 1))
        skip = self._plan((center + direction * 0.5) % 360.0, direction, arrive, bars, speed, min_w, depth - 1)
        return max(hit, skip)

    def _worth_hitting(self, bar, bars, theta, direction, speed, min_w):
        c = self.cfg
        others = [(b.center, b.width, b.kind) for b in bars
                  if b is not bar and (b.kind != "blue" or c["click_blue"])]
        dead = self.dead_time()
        hit = self._bar_value(bar.kind) + self._plan(
            (bar.center - direction * speed * dead) % 360.0, -direction, dead, others, speed, min_w, c["plan_depth"])
        skip = self._plan((bar.center + direction * 0.5) % 360.0, direction, 0.0, others, speed, min_w,
                          c["plan_depth"])
        return hit >= skip

    # outcome of a click

    def _resolve(self, t, outcome, flash_seen):
        c = self.cfg
        if outcome == "miss":
            self.misses += 1
            self.lockout_until = t + c["miss_lockout_ms"] / 1000.0
        else:
            self.hits += 1
        if flash_seen and c["auto_latency"]:
            # The flash is drawn on the game frame that registered the click,
            # so click frame -> flash frame is exactly the lead we aim with.
            lead = t - self.pending["t"]
            if 0.005 < lead < 0.15 and self.frame_dt * 1000.0 <= c["latency_learn_max_frame_ms"]:
                self.lat_samples.append(lead)
                if len(self.lat_samples) >= (3 if self.frame_dt < 0.015 else 5):
                    self.latency = float(np.median(self.lat_samples))
        self.pending = None

    def _update_pending(self, det, t, reversed_):
        # The pick turns around on hits and misses alike, so the flash colour
        # decides. At low frame rates the short hit flash can be missed; a
        # turnaround without a red flash after it then counts as a hit.
        p = self.pending
        if det.flash is not None:
            self._resolve(t, det.flash, flash_seen=True)
            return
        if reversed_ and p.get("reversed_at") is None:
            p["reversed_at"] = t
        if p.get("reversed_at") is not None and t - p["reversed_at"] >= max(0.04, 1.5 * self.frame_dt):
            self._resolve(t, "hit", flash_seen=False)
        elif t - p["t"] > self.cfg["hit_confirm_ms"] / 1000.0:
            self.pending = None

    def update(self, det, t, armed):
        """Returns (click, boost) for this frame."""
        cfg = self.cfg
        tracked, reversed_ = self._track(det)
        if self.pending is not None:
            self._update_pending(det, t, reversed_)

        if not (armed and tracked) or len(self.hist) < 3 or abs(self.omega) < 20:
            # Keep boosting through flashes and short dropouts so the pick
            # never has to spin up again.
            self.boost_on = cfg["boost_mode"] == "always" and armed and t - self.last_good_t < 1.5
            return False, self.boost_on
        self.last_good_t = t

        theta, speed = det.pick, abs(self.omega)
        direction = 1 if self.omega > 0 else -1
        steady = self._steady()
        if steady:
            self.speed_samples.append(speed)
        can_click = (self.pending is None and t >= self.lockout_until
                     and t - self.last_click_t >= cfg["min_click_interval_ms"] / 1000.0
                     and steady and self._speed_plausible(speed))
        min_w = self.min_width(speed)
        travel = speed * self.latency

        click = False
        for bar in det.bars:
            if click or not can_click or bar.width < min_w:
                continue
            if bar.kind == "blue" and not cfg["click_blue"]:
                continue
            half = bar.width / 2.0
            margin = min(cfg["edge_margin_deg"], half * 0.5)
            ahead = wrap180(bar.center - theta) * direction      # > 0: centre still ahead
            send_in = (ahead - travel) / speed                     # click now lands this early
            if not -(half - margin) / speed <= send_in < 0.5 * self.frame_dt:
                continue
            if cfg["strategy"] == "plan" and not self._worth_hitting(bar, det.bars, theta, direction, speed, min_w):
                self.skipped += 1
                continue
            click = True
            self.target = bar

        if click:
            self.clicks += 1
            self.last_click_t = t
            self.pending = {"t": t}
        self.boost_on = cfg["boost_mode"] == "always"
        return click, self.boost_on


user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE = 0
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
VK_LBUTTON, VK_RBUTTON = 0x01, 0x02


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("pad", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


def send_mouse(flags):
    inp = INPUT(type=INPUT_MOUSE)
    inp.u.mi = MOUSEINPUT(0, 0, 0, flags, 0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


class Mouse:
    """Left clicks run on a worker thread so the vision loop never sleeps."""

    def __init__(self, hold_ms):
        self.hold = hold_ms / 1000.0
        self.clicks = queue.Queue()
        self.right_down = False
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            self.clicks.get()
            send_mouse(MOUSEEVENTF_LEFTDOWN)
            time.sleep(self.hold)
            send_mouse(MOUSEEVENTF_LEFTUP)

    def click(self):
        self.clicks.put(None)

    def set_boost(self, on):
        if on != self.right_down:
            send_mouse(MOUSEEVENTF_RIGHTDOWN if on else MOUSEEVENTF_RIGHTUP)
            self.right_down = on


def make_dpi_aware():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        user32.SetProcessDPIAware()


def cursor_pos():
    p = wt.POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def foreground_title():
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(user32.GetForegroundWindow(), buf, 256)
    return buf.value


def key_down(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def beep(*tones):
    def play():
        for freq, ms in tones:
            winsound.Beep(freq, ms)
    threading.Thread(target=play, daemon=True).start()


class Hotkeys:
    KEYS = {"center": 0x74, "rim": 0x75, "toggle": 0x76, "debug": 0x77, "quit": 0x2E}  # F5 F6 F7 F8 DEL

    def __init__(self):
        self.down = {name: False for name in self.KEYS}

    def pressed(self):
        out = []
        for name, vk in self.KEYS.items():
            now = key_down(vk)
            if now and not self.down[name]:
                out.append(name)
            self.down[name] = now
        return out


class Capture:
    """Square screen region; dxcam (DXGI) when possible, mss otherwise."""

    def __init__(self, left, top, size, mode):
        self.size = size
        self.cam = None
        self.name = "mss"
        if mode in ("auto", "dxcam"):
            try:
                import dxcam
                screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
                if left >= 0 and top >= 0 and left + size <= screen_w and top + size <= screen_h:
                    self.cam = dxcam.create(output_idx=0, output_color="BGR")
                    self.region = (left, top, left + size, top + size)
                    self.name = "dxcam"
            except Exception as e:     # dxcam raises a variety of COM errors
                print(f"[i] dxcam unavailable ({e}), using mss")
        if self.cam is None:
            import mss
            self.sct = mss.mss()
            self.monitor = {"left": left, "top": top, "width": size, "height": size}

    def grab(self):
        """Returns (frame, timestamp), or (None, None) if the screen hasn't changed."""
        if self.cam is not None:
            frame = self.cam.grab(region=self.region)
            return (None, None) if frame is None else (frame, time.perf_counter())
        img = np.asarray(self.sct.grab(self.monitor))
        return np.ascontiguousarray(img[:, :, :3]), time.perf_counter()

    def close(self):
        if self.cam is not None:
            self.cam.release()
            self.cam = None


class Recorder:
    """Lossless video of the dial plus a per-frame CSV of what the bot saw and did.

    Encoding runs on a worker thread; frames it can't keep up with are dropped
    and get video_idx -1 in the CSV.
    """

    def __init__(self, cfg, size):
        folder = os.path.join(app_dir(), "recordings")
        os.makedirs(folder, exist_ok=True)
        self.base = os.path.join(folder, time.strftime("rec_%Y%m%d_%H%M%S"))
        self.writer = cv2.VideoWriter(self.base + ".mkv", cv2.VideoWriter_fourcc(*"FFV1"), 165, (size, size))
        if not self.writer.isOpened():
            self.writer = cv2.VideoWriter(self.base + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), 165, (size, size))
        self.csv = open(self.base + ".csv", "w", encoding="utf-8")
        self.csv.write("frame,video_idx,t,pick,score,omega,bot_on,armed,bot_click,lmb,rmb,timer,bars\n")
        with open(self.base + ".json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        self.frames = queue.Queue(maxsize=400)
        self.frame_no = self.video_idx = self.dropped = 0
        self.t0 = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while (frame := self.frames.get()) is not None:
            self.writer.write(frame)

    def add(self, frame, t, det, ctrl, bot_on, armed, click):
        if self.t0 is None:
            self.t0 = t
        video_idx = -1
        try:
            self.frames.put_nowait(np.ascontiguousarray(frame))
            video_idx = self.video_idx
            self.video_idx += 1
        except queue.Full:
            self.dropped += 1
        bars = ";".join(f"{b.kind[0]}:{b.start:.1f}:{b.width:.1f}" for b in det.bars)
        pick = "" if det.pick is None else f"{det.pick:.2f}"
        timer = "" if ctrl.time_left is None else round(ctrl.time_left, 2)
        self.csv.write(f"{self.frame_no},{video_idx},{t - self.t0:.5f},{pick},{det.score:.1f},{ctrl.omega:.1f},"
                       f"{int(bot_on)},{int(armed)},{int(click)},{int(key_down(VK_LBUTTON))},"
                       f"{int(key_down(VK_RBUTTON))},{timer},{bars}\n")
        self.frame_no += 1

    def stop(self):
        self.frames.put(None)
        self.thread.join()
        self.writer.release()
        self.csv.close()
        print(f"\n[i] recording saved: {self.base}.*  ({self.video_idx} frames, {self.dropped} dropped)")


def draw_debug(roi, det, ctrl, cfg, enabled):
    img = roi.copy()
    c = (img.shape[1] // 2, img.shape[0] // 2)
    r_in, r_out = int(cfg["band_inner"] * cfg["radius"]), int(cfg["band_outer"] * cfg["radius"])
    cv2.circle(img, c, r_in, (80, 80, 80), 1)
    cv2.circle(img, c, r_out, (80, 80, 80), 1)
    for b in det.bars:
        color = (0, 255, 255) if b.kind == "yellow" else (255, 128, 0)
        cv2.ellipse(img, c, (r_out + 6, r_out + 6), 0, b.start, b.start + b.width, color, 4)

    def ray(angle, color, thickness):
        a = math.radians(angle)
        p0 = (int(c[0] + r_in * math.cos(a)), int(c[1] + r_in * math.sin(a)))
        p1 = (int(c[0] + (r_out + 14) * math.cos(a)), int(c[1] + (r_out + 14) * math.sin(a)))
        cv2.line(img, p0, p1, color, thickness)

    if det.pick is not None:
        ray(det.pick, (0, 0, 255), 2)
        ray((det.pick + ctrl.omega * ctrl.latency) % 360, (0, 255, 0), 1)
    text = (f"{'ON' if enabled else 'OFF'}  w={ctrl.omega:6.0f}deg/s  lat={ctrl.latency * 1000:4.1f}ms  "
            f"hit={ctrl.hits} miss={ctrl.misses}")
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


BANNER = """
=====================  PICK THE LOCK BOT  =====================
 Open the Pick The Lock minigame in Dota 2 (Borderless Window
 recommended), then:
   F5        hover the mouse over the CENTRE of the lock dial
   F6        hover the mouse over the OUTER EDGE of the dark ring
             (the outer tip of a yellow bar)
   F7        start / stop
   F8        debug view + record what the bot sees (recordings/)
   DEL       quit
 Calibration is saved to config.json and reused next time.
================================================================
"""


class Bot:
    STATUS_EVERY = 0.5
    TIMER_EVERY = 0.045
    DEBUG_EVERY = 0.025

    def __init__(self):
        self.cfg, self.user_keys = load_config()
        self.mouse = Mouse(self.cfg["click_hold_ms"])
        self.ctrl = Controller(self.cfg)
        self.hotkeys = Hotkeys()
        self.capture = self.detector = self.timer = None
        self.recorder = None
        self.enabled = False
        self.debug = False
        self.pending_center = self.cfg["center"]
        self.timer_value = self.timer_seen_at = None
        self.last_timer_read = self.last_debug_draw = 0.0
        self.frames = self.frames_with_pick = 0
        self.stats_since = time.perf_counter()

    def run(self):
        print(BANNER)
        print(f"[i] config: {CONFIG_PATH}")
        print(f"[i] fps profile: {self.cfg['fps_profile']}"
              + (f" (starting as {self.cfg['active_profile']})" if self.cfg["fps_profile"] == "auto" else ""))
        if self.cfg["center"] and self.cfg["radius"]:
            self.start_capture()
        else:
            print("[i] not calibrated yet - press F5 then F6 over the dial")
        try:
            while self.handle_hotkeys():
                if self.capture is None:
                    time.sleep(0.02)
                    continue
                frame, t = self.capture.grab()
                if frame is None:
                    time.sleep(0.0005)
                    continue
                self.step(frame, t)
        finally:
            self.shutdown()

    def start_capture(self):
        if self.capture is not None:
            self.capture.close()
        (cx, cy), radius = self.cfg["center"], self.cfg["radius"]
        half = int(math.ceil(radius * 1.08))
        self.capture = Capture(int(cx) - half, int(cy) - half, 2 * half, self.cfg["capture"])
        self.detector = Detector(self.cfg, radius)
        self.timer = TimerReader(radius) if self.cfg["read_timer"] else None
        self.ctrl.reset()
        print(f"[i] dial centre=({cx},{cy}) radius={radius:.0f}px  capture={self.capture.name}")

    def handle_hotkeys(self):
        """Returns False when the user asked to quit."""
        for key in self.hotkeys.pressed():
            if key == "quit":
                return False
            if key == "center":
                self.pending_center = list(cursor_pos())
                print(f"[i] centre set to {self.pending_center} - now hover the ring edge and press F6")
            elif key == "rim":
                self.calibrate_radius()
            elif key == "toggle":
                self.toggle()
            elif key == "debug":
                self.toggle_debug()
        return True

    def calibrate_radius(self):
        if not self.pending_center:
            print("[!] press F5 over the centre first")
            return
        x, y = cursor_pos()
        radius = math.hypot(x - self.pending_center[0], y - self.pending_center[1])
        if radius < 40:
            print("[!] radius too small - hover the OUTER edge of the dark ring")
            return
        self.cfg["center"], self.cfg["radius"] = self.pending_center, radius
        save_config(self.cfg, self.user_keys)
        print("[i] calibration saved")
        self.start_capture()

    def toggle(self):
        if self.capture is None:
            print("[!] calibrate first (F5, F6)")
            return
        self.enabled = not self.enabled
        self.ctrl.reset()
        self.detector.reset()
        self.mouse.set_boost(False)
        if self.enabled and self.cfg["move_cursor_on_start"]:
            user32.SetCursorPos(int(self.cfg["center"][0]), int(self.cfg["center"][1]))
        print(f"\n[i] bot {'ENABLED' if self.enabled else 'disabled'}")
        beep((1200, 120)) if self.enabled else beep((400, 250))

    def toggle_debug(self):
        self.debug = not self.debug
        if self.debug and self.capture is not None:
            self.recorder = Recorder(self.cfg, self.capture.size)
            print(f"\n[i] debug view on, recording to {self.recorder.base}.mkv (F8 again to stop)")
            beep((800, 60), (800, 60))
        elif not self.debug:
            cv2.destroyAllWindows()
            if self.recorder is not None:
                self.recorder.stop()
                self.recorder = None
                beep((600, 60), (600, 60))

    def step(self, frame, t):
        ctrl = self.ctrl
        predicted = ctrl.predicted_angle(t)
        det = self.detector.process(frame, t, predicted,
                                    ctrl.search_window() if predicted is not None else None, ctrl.moving())
        armed = self.enabled and (not self.cfg["require_dota_foreground"] or "Dota 2" in foreground_title())
        ctrl.time_left = self.read_timer(frame, t)
        click, boost = ctrl.update(det, t, armed)
        if click:
            self.mouse.click()
        self.mouse.set_boost(boost and armed)
        if self.recorder is not None:
            self.recorder.add(frame, t, det, ctrl, self.enabled, armed, click)

        self.frames += 1
        self.frames_with_pick += det.pick is not None
        now = time.perf_counter()
        if now - self.stats_since >= self.STATUS_EVERY:
            self.print_status(det, self.frames / (now - self.stats_since))
            self.frames = self.frames_with_pick = 0
            self.stats_since = now
        if self.debug and t - self.last_debug_draw >= self.DEBUG_EVERY:
            self.last_debug_draw = t
            cv2.imshow("Pick The Lock - debug", draw_debug(frame, det, ctrl, self.cfg, self.enabled))
            cv2.waitKey(1)

    def read_timer(self, frame, t):
        if self.timer is not None and t - self.last_timer_read >= self.TIMER_EVERY:
            self.last_timer_read = t
            value = self.timer.read(frame)
            if value is not None:
                self.timer_value, self.timer_seen_at = value, t
        # extrapolate between reads, give up after a second without one
        if self.timer_seen_at is None or t - self.timer_seen_at > 1.0:
            return None
        return max(0.0, self.timer_value - (t - self.timer_seen_at))

    def print_status(self, det, fps):
        cfg = self.cfg
        game_visible = self.frames_with_pick > 0.7 * self.frames
        if game_visible and cfg["fps_profile"] == "auto":
            wanted = "120" if fps >= 70 else "30"
            if wanted != cfg["active_profile"]:
                apply_profile(cfg, wanted, self.user_keys)
                self.ctrl.load_latency()
                print(f"\n[i] game redraws ~{fps:.0f} fps -> fps profile {wanted} "
                      f"(latency {self.ctrl.latency * 1000:.0f} ms)")
        elif game_visible and self.enabled and fps < 0.6 * cfg["expected_fps"]:
            print(f"\n[!] game redraws only ~{fps:.0f} fps but fps_profile is {cfg['active_profile']} "
                  f"- set \"fps_profile\": \"30\" or \"auto\"")
        ctrl = self.ctrl
        pick = "  ----" if det.pick is None else f"{det.pick:6.1f}"
        yellow = sum(b.kind == "yellow" for b in det.bars)
        timer = "--.-" if ctrl.time_left is None else f"{ctrl.time_left:4.1f}"
        print(f"\r[{'ON ' if self.enabled else 'OFF'}] fps={fps:5.1f}/{cfg['active_profile']} pick={pick}"
              f" w={ctrl.omega:6.0f} bars(y/b)={yellow}/{len(det.bars) - yellow}"
              f" lat={ctrl.latency * 1000:4.1f}ms timer={timer}"
              f" clicks={ctrl.clicks} hits={ctrl.hits} miss={ctrl.misses}   ", end="", flush=True)

    def shutdown(self):
        self.mouse.set_boost(False)
        if self.recorder is not None:
            self.recorder.stop()
        if self.cfg["auto_latency"] and len(self.ctrl.lat_samples) >= 5:
            self.cfg[latency_key(self.cfg["active_profile"])] = round(self.ctrl.latency * 1000, 1)
            save_config(self.cfg, self.user_keys)
        if self.capture is not None:
            self.capture.close()
        cv2.destroyAllWindows()
        print("\n[i] bye")


def main():
    make_dpi_aware()
    ctypes.windll.winmm.timeBeginPeriod(1)
    Bot().run()


if __name__ == "__main__":
    main()
