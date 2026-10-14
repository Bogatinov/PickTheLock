"""Whole-game simulator for strategy work.

Runs the bot's real Controller (tracking, click timing, outcome handling,
planner) against a model of the minigame, with perfect detections instead
of rendered frames. The model is fitted to recorded games:

  * pick ~128 deg/s with boost held; it reverses on every hit and miss
  * bars start ~22 deg wide and shrink 6.2 deg/s
  * 30 s on the clock, a blue bar adds 1.5 s, both colours score 1000
  * more bars appear as the game goes on (by elapsed time)
  * hit flash ~30 ms, miss flash ~250 ms, a miss disables the pick for 0.5 s

Usage (from the repository root):
    python -m tools.strategy_sim [games] [fps]
"""
import random
import sys

import numpy as np

import picklock as pl

FPS = 165
SPEED = 128.0
LATENCY = 0.025
JITTER = 0.004


class Game:
    def __init__(self, seed, ramp="time"):
        self.rng = random.Random(seed)
        self.ramp = ramp
        self.t = 0.0
        self.clock = 30.0
        self.theta = self.rng.uniform(0, 360)
        self.direction = 1
        self.bars = []                 # {"c": centre, "w": width, "kind": ...}
        self.score = 0
        self.hits = self.blue_hits = self.misses = 0
        self.disabled_until = 0.0
        self.flash = (None, -1.0)
        self.flash_prob = 1.0          # chance a hit flash lands on a rendered frame

    def spawn_rates(self):
        # Fitted to recordings: on the dial at 0-5 / 20-25 / 40-45 s there are
        # ~1.0 / 1.5 / 4.0 yellow and ~0.05 / 0.6 / 0.9 blue bars.
        e = self.t if self.ramp == "time" else self.hits * 1.1
        return 0.60 + 0.0006 * e * e, min(0.45, 0.03 + 0.015 * e)

    def spawn(self, kind):
        for _ in range(20):
            c = self.rng.uniform(0, 360)
            if abs(pl.wrap180(c - self.theta)) > 30 and all(abs(pl.wrap180(c - b["c"])) > 25 for b in self.bars):
                self.bars.append({"c": c, "w": 22.0, "kind": kind})
                return

    def step(self, dt):
        self.t += dt
        self.clock -= dt
        self.theta = (self.theta + self.direction * SPEED * dt) % 360
        for b in self.bars:
            b["w"] -= 6.2 * dt
        self.bars = [b for b in self.bars if b["w"] > 1.0]
        yellow_rate, blue_rate = self.spawn_rates()
        if self.rng.random() < yellow_rate * dt:
            self.spawn("yellow")
        if self.rng.random() < blue_rate * dt:
            self.spawn("blue")

    def click(self):
        if self.t < self.disabled_until:
            return
        for b in self.bars:
            if abs(pl.wrap180(self.theta - b["c"])) <= b["w"] / 2:
                self.bars.remove(b)
                self.hits += 1
                self.score += 1000
                if b["kind"] == "blue":
                    self.blue_hits += 1
                    self.clock += 1.5
                self.direction *= -1
                self.flash = ("hit" if self.rng.random() < self.flash_prob else "hit-unseen", self.t)
                return
        self.misses += 1
        self.direction *= -1
        self.disabled_until = self.t + 0.5
        self.flash = ("miss", self.t)

    def detection(self, rng):
        bars = [pl.Bar((b["c"] - b["w"] / 2) % 360, b["w"], b["kind"]) for b in self.bars]
        kind, started = self.flash
        flash = None
        if kind == "hit" and 0 <= self.t - started < 0.03:
            flash = "hit"
        elif kind == "miss" and 0 <= self.t - started < 0.25:
            flash = "miss"
        pick = (self.theta + rng.gauss(0, 0.2)) % 360
        return pl.Detection(self.t, pick, 100.0, bars, flash is not None, flash)


def play(seed, strategy="plan", ramp="time", fps=FPS, profile="120", flash_prob=1.0, **overrides):
    """Returns (score, length, hits, blue hits, misses, skips).

    The game registers clicks only on its own frames, and the bot only sees
    those frames - like Dota running with fps_max_ui = fps.
    """
    cfg = pl.make_config(profile, boost_mode="always", strategy=strategy)
    cfg.update(overrides)
    game = Game(seed, ramp)
    game.flash_prob = flash_prob
    ctrl = pl.Controller(cfg)
    rng = random.Random(seed + 999)
    dt = 1.0 / fps
    substeps = max(1, int(round(4 * FPS / fps)))
    in_flight = []
    while game.clock > 0:
        for _ in range(substeps):
            game.step(dt / substeps)
        for arrival in [a for a in in_flight if a <= game.t]:
            game.click()
            in_flight.remove(arrival)
        ctrl.time_left = game.clock if cfg["read_timer"] else None
        click, _ = ctrl.update(game.detection(rng), game.t, True)
        if click:
            in_flight.append(game.t + LATENCY + rng.uniform(-JITTER, JITTER))
    return game.score, game.t, game.hits, game.blue_hits, game.misses, ctrl.skipped


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else FPS
    profile = "120" if fps >= 70 else "30"
    for strategy in ("greedy", "plan"):
        r = np.array([play(s, strategy, fps=fps, profile=profile) for s in range(games)])
        print(f"{strategy:6s} @ {fps} fps: score avg {r[:, 0].mean():6.0f} (+-{r[:, 0].std() / np.sqrt(games):4.0f})"
              f"  median {np.median(r[:, 0]):6.0f}  game {r[:, 1].mean():4.1f}s"
              f"  hits {r[:, 2].mean():4.1f}  blue {r[:, 3].mean():4.1f}  misses {r[:, 4].mean():.2f}")


if __name__ == "__main__":
    main()
