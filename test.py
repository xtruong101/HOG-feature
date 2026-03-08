"""
test.py  –  Nhận diện người bằng HOG + KNN + Sliding Window
============================================================
Yêu cầu : đã chạy train.py để có knn_hog_model.pkl

Nguồn ảnh (--source)
---------------------
  dataset  : ảnh từ split test của INRIA Person trên HuggingFace (mặc định)
  image    : 1 ảnh đơn lẻ    --input path/to/img.jpg
  folder   : cả thư mục      --input path/to/folder/

Ví dụ
-----
  python test.py
  python test.py --source dataset --n 10
  python test.py --source image  --input photo.jpg
  python test.py --source folder --input ./test_images/
  python test.py --conf 0.7 --stride 16 --iou 0.25
"""

import os
import sys
import glob
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")            # đổi thành "Agg" nếu không có GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import (
    WIN_H, WIN_W,
    SCALES, STRIDE, MAX_DIM,
    CONF_THRESHOLD, IOU_THRESHOLD,
    MODEL_PATH, OUTPUT_DIR,
    extract_hog, pil_to_cv2,
    sliding_window, nms,
    draw_detections, save_result, load_model
)


# ============================================================
# 1. LOAD ẢNH TEST
# ============================================================

def _get_col(column_names: list, candidates: list, fallback: int = 0):
    """Trả về tên cột đầu tiên tìm thấy trong candidates."""
    for c in candidates:
        if c in column_names:
            return c
    return column_names[fallback]


def load_from_dataset(n: int = 10) -> tuple:
    """Lấy n ảnh có người (label=1) từ split test của INRIA Person."""
    from datasets import load_dataset

    print("[INFO] Đang tải ảnh test từ Hugging Face ...")
    ds         = load_dataset("marcelarosalesj/inria-person")
    split_name = "test" if "test" in ds else list(ds.keys())[-1]
    split      = ds[split_name]

    col_img = _get_col(split.column_names, ["image", "img"])
    col_lbl = _get_col(split.column_names, ["label", "labels"], -1) \
              if any(c in split.column_names
                     for c in ["label", "labels"]) else None

    images, names = [], []
    for item in split:
        if col_lbl and int(item[col_lbl]) != 1:
            continue
        images.append(pil_to_cv2(item[col_img]))
        names.append(f"dataset_{len(images):03d}")
        if len(images) >= n:
            break

    print(f"[INFO] Lấy được {len(images)} ảnh từ split '{split_name}'.")
    return images, names


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_from_image(path: str) -> tuple:
    """Đọc 1 ảnh từ đường dẫn file."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    name = os.path.splitext(os.path.basename(path))[0]
    return [img], [name]


def load_from_folder(folder: str) -> tuple:
    """Đọc tất cả ảnh (không đệ quy) trong thư mục."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Không tìm thấy thư mục: {folder}")

    paths = sorted([
        p for p in glob.glob(os.path.join(folder, "*"))
        if os.path.splitext(p)[1].lower() in SUPPORTED
    ])
    if not paths:
        raise ValueError(f"Không có ảnh hỗ trợ trong: {folder}")

    images, names = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)
            names.append(os.path.splitext(os.path.basename(p))[0])

    print(f"[INFO] Đọc được {len(images)} ảnh từ: {folder}")
    return images, names


# ============================================================
# 2. PHÁT HIỆN NGƯỜI (SLIDING WINDOW + SCALE PYRAMID)
# ============================================================

def detect_persons(image_bgr: np.ndarray,
                   knn,
                   scaler,
                   conf_threshold: float = CONF_THRESHOLD,
                   stride: int = STRIDE,
                   scales: list = None,
                   verbose: bool = False) -> list:
    """
    Multi-scale sliding window + batch KNN prediction.

    Trả về list of (x0, y0, x1, y1, confidence) trên ảnh gốc.
    """
    if scales is None:
        scales = SCALES

    h0, w0 = image_bgr.shape[:2]

    # Tự động resize nếu ảnh quá lớn
    if max(h0, w0) > MAX_DIM:
        ratio     = MAX_DIM / max(h0, w0)
        new_w     = max(WIN_W, int(w0 * ratio))
        new_h     = max(WIN_H, int(h0 * ratio))
        image_bgr = cv2.resize(image_bgr, (new_w, new_h))
        h0, w0    = image_bgr.shape[:2]
        if verbose:
            print(f"    [resize] → {w0}×{h0}px")

    detections = []

    for scale in scales:
        new_h   = max(WIN_H, int(h0 * scale))
        new_w   = max(WIN_W, int(w0 * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h))

        # Thu thập tất cả cửa sổ trong scale này
        coords, feats = [], []
        for x, y, patch in sliding_window(resized, stride):
            coords.append((x, y))
            feats.append(extract_hog(patch))

        if not feats:
            continue

        # Batch predict: 1 lần cho cả scale → nhanh hơn nhiều
        X_batch = scaler.transform(np.array(feats, dtype=np.float32))
        probas  = knn.predict_proba(X_batch)            # (N, 2)

        scale_detections = 0
        for (x, y), proba in zip(coords, probas):
            if len(proba) < 2:
                continue
            conf = float(proba[1])
            if conf >= conf_threshold:
                x0 = max(0, min(int(x           / scale), w0))
                y0 = max(0, min(int(y           / scale), h0))
                x1 = max(0, min(int((x + WIN_W) / scale), w0))
                y1 = max(0, min(int((y + WIN_H) / scale), h0))
                detections.append((x0, y0, x1, y1, conf))
                scale_detections += 1

        if verbose:
            print(f"      scale={scale:.2f}: {scale_detections} detections")

    return detections


# ============================================================
# 3. XỬ LÝ DANH SÁCH ẢNH
# ============================================================

def process_images(images: list, names: list,
                   knn, scaler,
                   conf_threshold: float,
                   iou_threshold: float,
                   stride: int,
                   scales: list,
                   out_dir: str,
                   verbose: bool = False) -> list:
    """Detect + NMS + lưu file cho từng ảnh."""
    os.makedirs(out_dir, exist_ok=True)
    results      = []
    total_boxes  = 0

    for idx, (img, name) in enumerate(zip(images, names)):
        h, w = img.shape[:2]
        print(f"  [{idx+1:3d}/{len(images)}] {name}  ({w}×{h}px) ...",
              end="  ", flush=True)

        raw   = detect_persons(img, knn, scaler, conf_threshold, stride,
                               scales=scales, verbose=verbose)
        final = nms(raw, iou_threshold)
        total_boxes += len(final)

        print(f"raw={len(raw):4d}  →  NMS={len(final):3d}")

        out_img  = draw_detections(img, final)
        out_path = save_result(out_img, f"{name}_result.jpg", out_dir)
        if verbose:
            print(f"    [lưu] {out_path}")
        results.append((out_img, name, len(final)))

    print(f"\n[INFO] Tổng bounding box sau NMS: {total_boxes}")
    return results


# ============================================================
# 4. HIỂN THỊ LƯỚI KẾT QUẢ
# ============================================================

def show_grid(results: list, out_dir: str) -> None:
    n    = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(6 * cols, 5 * rows),
                             squeeze=False)
    flat = axes.flatten()

    for i, (img_bgr, name, n_det) in enumerate(results):
        flat[i].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        flat[i].set_title(f"{name}\n({n_det} người)", fontsize=9)
        flat[i].axis("off")

    for j in range(len(results), len(flat)):
        flat[j].axis("off")

    legend = mpatches.Patch(facecolor="lime", edgecolor="k",
                            label="Person (detected)")
    fig.legend(handles=[legend], loc="lower center",
               fontsize=11, ncol=1)
    plt.suptitle("Kết quả nhận diện người  –  HOG + KNN + Sliding Window",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    grid_path = os.path.join(out_dir, "detection_grid.png")
    plt.savefig(grid_path, dpi=120)
    plt.show()
    print(f"[INFO] Lưới kết quả lưu tại: {grid_path}")


# ============================================================
# CLI ARGUMENTS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Nhận diện người: HOG + KNN + Sliding Window"
    )
    p.add_argument("--source",
                   choices=["dataset", "image", "folder"],
                   default="dataset",
                   help="Nguồn ảnh (mặc định: dataset)")
    p.add_argument("--input", default=None,
                   help="Đường dẫn ảnh / thư mục "
                        "(bắt buộc khi --source=image hoặc folder)")
    p.add_argument("--n", type=int, default=10,
                   help="Số ảnh lấy từ dataset (mặc định: 10)")
    p.add_argument("--model", default=MODEL_PATH,
                   help=f"File model pickle (mặc định: {MODEL_PATH})")
    p.add_argument("--output", default=OUTPUT_DIR,
                   help=f"Thư mục lưu kết quả (mặc định: {OUTPUT_DIR})")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                   help=f"Ngưỡng confidence (mặc định: {CONF_THRESHOLD})")
    p.add_argument("--iou", type=float, default=IOU_THRESHOLD,
                   help=f"Ngưỡng IoU / NMS (mặc định: {IOU_THRESHOLD})")
    p.add_argument("--stride", type=int, default=STRIDE,
                   help=f"Bước trượt px (mặc định: {STRIDE})")
    p.add_argument("--scales", type=float, nargs="+", default=SCALES,
                   help=f"Danh sách scale (mặc định: {SCALES})")
    p.add_argument("--verbose", action="store_true",
                   help="In chi tiết thông tin nhận diện")
    p.add_argument("--no-show", action="store_true",
                   help="Không mở cửa sổ matplotlib")
    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # --- Load model ---
    knn, scaler = load_model(args.model)

    # --- Load ảnh ---
    if args.source == "dataset":
        images, names = load_from_dataset(n=args.n)

    elif args.source == "image":
        if not args.input:
            print("[LỖI] --source=image cần --input <file ảnh>")
            sys.exit(1)
        images, names = load_from_image(args.input)

    else:   # folder
        if not args.input:
            print("[LỖI] --source=folder cần --input <thư mục>")
            sys.exit(1)
        images, names = load_from_folder(args.input)

    # --- Detect ---
    print(f"\n[INFO] Nhận diện  "
          f"conf≥{args.conf}  stride={args.stride}px  IoU≤{args.iou}")
    print(f"       scales={args.scales}")
    results = process_images(
        images, names, knn, scaler,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        stride=args.stride,
        scales=args.scales,
        out_dir=args.output,
        verbose=args.verbose
    )

    # --- Hiển thị ---
    if not args.no_show:
        show_grid(results, args.output)

    print(f"\n[HOÀN THÀNH] Ảnh kết quả tại: ./{args.output}/")


if __name__ == "__main__":
    main()
