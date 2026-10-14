"""End-to-end test on synthetic frames.

Renders a lock dial (fading bars, a glowing pick, hit/miss flashes, sensor
noise) and runs the real Detector + Controller on it, with input latency and
game-frame quantisation like the real game.

Usage (from the repository root):
    python -m tools.vision_sim [seconds]
"""
import math
import random
import sys

import cv2
import numpy as np

from picklock import Controller, Detector, make_config, wrap180

FPS = 165
R = 220
SIZE = int(math.ceil(R * 1.08)) * 2


class Game:
    def __init__(self, seed, speed=160.0):
        self.rng = random.Random(seed)
        self.theta, self.direction, self.speed = 0.0, 1, speed
        self.disabled_until = 0.0
        self.score = self.hits = self.misses = 0
        self.flash = (None, -1.0)
        self.flash_prob = 1.0
        self.bars = []
        self.bars = [self.new_bar("yellow", 0)]
        self.bars.append(self.new_bar("blue", 0))

    def new_bar(self, kind, t):
        while True:                    # keep bars apart from each other and from the pick
            c = self.rng.uniform(0, 360)
            if abs(wrap180(c - self.theta)) > 40 and all(abs(wrap180(c - b["center"])) > 40 for b in self.bars):
                return {"kind": kind, "center": c, "born": t}

    @staticmethod
    def width(bar, t):
        return 30.0 - 6.0 * (t - bar["born"])

    def expire(self, t):
        for i, b in enumerate(self.bars):
            if self.width(b, t) <= 1.5:
                del self.bars[i]
                self.bars.insert(i, self.new_bar(b["kind"], t))

    def step(self, dt, boost):
        self.theta = (self.theta + self.direction * self.speed * (3.0 if boost else 1.0) * dt) % 360

    def click(self, t):
        if t < self.disabled_until:
            return
        for i, b in enumerate(self.bars):
            if abs(wrap180(self.theta - b["center"])) <= self.width(b, t) / 2:
                self.hits += 1
                self.score += 1000
                self.direction *= -1
                self.speed *= 1.04
                self.flash = ("hit" if self.rng.random() < self.flash_prob else "hit-unseen", t)
                del self.bars[i]
                self.bars.insert(i, self.new_bar(b["kind"], t))
                return
        self.misses += 1
        self.disabled_until = t + 0.5
        self.flash = ("miss", t)

    def render(self, t, noise):
        img = np.full((SIZE, SIZE, 3), 20, np.uint8)
        c = (SIZE // 2, SIZE // 2)
        cv2.circle(img, c, R, (38, 36, 34), -1)
        for b in self.bars:
            w = self.width(b, t)
            color = (40, 185, 225) if b["kind"] == "yellow" else (225, 140, 50)
            alpha = min(1.0, max(0.12, w / 30.0))       # bars fade as they shrink
            layer = img.copy()
            cv2.ellipse(layer, c, (int(R * 0.96), int(R * 0.96)), 0, b["center"] - w / 2, b["center"] + w / 2, color, -1)
            img = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
        cv2.circle(img, c, int(R * 0.53), (62, 60, 58), -1)
        a = math.radians(self.theta)
        p0 = (int(c[0] + R * 0.55 * math.cos(a)), int(c[1] + R * 0.55 * math.sin(a)))
        p1 = (int(c[0] + R * 0.97 * math.cos(a)), int(c[1] + R * 0.97 * math.sin(a)))
        glow = np.zeros_like(img)
        cv2.line(glow, p0, p1, (255, 90, 120), 9, cv2.LINE_AA)
        img = cv2.add(img, cv2.GaussianBlur(glow, (0, 0), 3))
        cv2.line(img, p0, p1, (255, 210, 220), 2, cv2.LINE_AA)
        kind, started = self.flash
        if kind == "hit" and 0 <= t - started < 0.03:
            img = cv2.add(img, np.full_like(img, (40, 25, 25)))
        elif kind == "miss" and 0 <= t - started < 0.25:
            img = (img * np.array([0.25, 0.25, 1.0])).astype(np.uint8)
        if noise:
            img = cv2.add(img, np.random.randint(0, noise, img.shape, np.uint8))
        return img


def run(seed, latency, jitter, boost_mode="off", seconds=60, fps=FPS, profile="120", flash_prob=1.0, speed=160.0):
    """The game redraws and registers clicks at `fps`, like fps_max_ui."""
    cfg = make_config(profile, boost_mode=boost_mode)
    game = Game(seed, speed)
    game.flash_prob = flash_prob
    det = Detector(cfg, R)
    ctrl = Controller(cfg)
    in_flight, t, dt = [], 0.0, 1.0 / fps
    boost = False
    substeps = max(1, int(round(8 * FPS / fps)))
    while t < seconds:
        for _ in range(substeps):
            game.step(dt / substeps, boost)
            game.expire(t)
            t += dt / substeps
        for arrival in [a for a in in_flight if a <= t]:
            game.click(t)
            in_flight.remove(arrival)
        frame = game.render(t, noise=12)
        d = det.process(frame, t, ctrl.predicted_angle(t), ctrl.search_window(), ctrl.moving())
        click, boost = ctrl.update(d, t, True)
        if click:
            in_flight.append(t + latency + random.uniform(0, jitter))
    return game, ctrl


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20
    for fps, profile, flash_prob in ((165, "120", 1.0), (30, "30", 0.5)):
        for seed in (1, 2):
            game, ctrl = run(seed, 0.022, 0.006, "always", seconds, fps, profile, flash_prob, speed=42.0)
            print(f"{fps:3d} fps seed {seed}: clicks {ctrl.clicks}  hits {game.hits}  misses {game.misses}"
                  f"  learned latency {ctrl.latency * 1000:.1f} ms")


if __name__ == "__main__":
    main()
