import argparse
import os
import cv2
import face_alignment
import numpy as np
import torch
from face_alignment.api import LandmarksType
from tqdm import tqdm


def _get_landmarks_type_2d() -> LandmarksType:
    lt = getattr(LandmarksType, "_2D", None)
    if lt is None:
        lt = getattr(LandmarksType, "TWO_D", None)
    if lt is None:
        for k in dir(LandmarksType):
            if "2" in k and "D" in k and not k.startswith("_"):
                lt = getattr(LandmarksType, k)
                break
    if lt is None:
        raise RuntimeError("face_alignment の LandmarksType に 2D が見つかりません。")
    return lt


def extract_68pts_to_npy(
    video_path: str,
    out_npy: str,
    every_n: int = 1,
    max_frames: int | None = None,
    verbose: bool = True,
):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lt2d = _get_landmarks_type_2d()
    fa = face_alignment.FaceAlignment(lt2d, flip_input=False, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    landmarks = []
    last = None

    if verbose:
        print(f"[info] device={device}, total_frames={total}, every_n={every_n}")

    pbar = (
        tqdm(total=total, desc=f"landmarks:{os.path.basename(video_path)}")
        if verbose
        else None
    )

    fidx = 0
    kept = 0
    while True:
        if max_frames is not None and fidx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if fidx % every_n != 0:
            fidx += 1
            if pbar:
                pbar.update(1)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(rgb)

        if preds is not None and len(preds) > 0:
            sel = None
            max_area = -1
            for p in preds:
                x1, y1 = p.min(axis=0)
                x2, y2 = p.max(axis=0)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    sel = p
            pts = sel.astype(np.float32)
            last = pts
        else:
            if last is not None:
                pts = last
            else:
                pts = np.zeros((68, 2), dtype=np.float32)

        landmarks.append(pts)
        kept += 1
        fidx += 1
        if pbar:
            pbar.update(1)

    cap.release()
    if pbar:
        pbar.close()

    if kept == 0:
        raise RuntimeError("フレームが読み込めませんでした。")

    L = np.stack(landmarks, axis=0)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, L)
    print(f"[OK] landmarks saved -> {out_npy}  shape={L.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_video", type=str)
    ap.add_argument("out_npy", type=str)
    ap.add_argument("--every_n", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    extract_68pts_to_npy(
        args.input_video, args.out_npy, every_n=args.every_n, max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
