"""Replays a recording through the current Detector + Controller.

Every click the bot would make is saved as an annotated frame, which makes it
easy to see what the bot was looking at.

Usage (from the repository root):
    python -m tools.replay recordings/rec_YYYYmmdd_HHMMSS [out_dir]
"""
import csv
import json
import os
import sys

import cv2

import picklock as pl


def main():
    base = sys.argv[1].removesuffix(".csv").removesuffix(".mkv")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else base + "_replay"
    os.makedirs(out_dir, exist_ok=True)

    with open(base + ".csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(base + ".json", encoding="utf-8") as f:
        recorded = json.load(f)
    cfg = pl.make_config(boost_mode="always", center=recorded["center"], radius=recorded["radius"])
    det = pl.Detector(cfg, cfg["radius"])
    ctrl = pl.Controller(cfg)
    row_of_frame = {int(r["video_idx"]): i for i, r in enumerate(rows) if r["video_idx"] != "-1"}

    video = cv2.VideoCapture(base + ".mkv")
    clicks, real_clicks = [], [float(r["t"]) for r in rows if r["bot_click"] == "1"]
    frame_no = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        row = rows[row_of_frame[frame_no]]
        frame_no += 1
        t = float(row["t"])
        predicted = ctrl.predicted_angle(t)
        d = det.process(frame, t, predicted, ctrl.search_window() if predicted is not None else None, ctrl.moving())
        ctrl.time_left = float(row["timer"]) if row.get("timer") else None
        click, _ = ctrl.update(d, t, True)
        if click:
            clicks.append(t)
            img = pl.draw_debug(frame, d, ctrl, cfg, True)
            target = ctrl.target
            cv2.putText(img, f"t={t:.2f} {target.kind} c={target.center:.1f} w={target.width:.1f}", (6, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out_dir, f"click_{len(clicks):03d}_t{t:06.2f}.png"), img)

    same = sum(1 for rc in real_clicks if any(abs(c - rc) < 0.05 for c in clicks))
    print(f"{frame_no} frames: replay clicks {len(clicks)}, recorded bot clicks {len(real_clicks)}, "
          f"matching {same}. Annotated frames in {out_dir}")


if __name__ == "__main__":
    main()
