"""
utils.py  –  Hàm & cấu hình dùng chung cho train.py và test.py
===============================================================
pip install numpy opencv-python scikit-image scikit-learn datasets pillow matplotlib
"""
import os
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog

# ============================================================
# CẤU HÌNH CHUNG  (chỉnh tại đây, áp dụng cho cả train & test)
# ============================================================

WIN_H          = 128    # chiều cao cửa sổ HOG (px)
WIN_W          = 64     # chiều rộng cửa sổ HOG (px)
CELL_SIZE      = 8      # kích thước cell (px)
BLOCK_SIZE     = 2      # số cell/chiều trong 1 block
NUM_BINS       = 9      # số bin histogram
STRIDE         = 32     # bước trượt sliding window (px)
SCALES         = [1.0, 0.7, 0.4]   # scale pyramid
MAX_DIM        = 800    # resize ảnh nếu cạnh dài > MAX_DIM trước khi detect
K_NEIGHBORS    = 7      # k trong KNN
IOU_THRESHOLD  = 0.3    # ngưỡng IoU cho NMS
CONF_THRESHOLD = 0.75   # ngưỡng xác suất để nhận là "người"
MAX_NEG_CROPS  = 15     # số patch cắt ngẫu nhiên từ mỗi ảnh nền
RANDOM_SEED    = 42
MODEL_PATH     = "knn_hog_model.pkl"
OUTPUT_DIR     = "detection_results"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# HOG FEATURE EXTRACTION
# ============================================================

def extract_hog(image: np.ndarray) -> np.ndarray:
    """
    Trích xuất HOG từ ảnh đầu vào.
    Tự động resize về WIN_W x WIN_H và chuyển sang xám.

    Trả về vector 1-D numpy float32 (3780 chiều với config mặc định).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.resize(gray, (WIN_W, WIN_H))

    feat = hog(
        gray,
        orientations=NUM_BINS,
        pixels_per_cell=(CELL_SIZE, CELL_SIZE),
        cells_per_block=(BLOCK_SIZE, BLOCK_SIZE),
        block_norm='L2-Hys',
        feature_vector=True
    )
    return feat.astype(np.float32)


# ============================================================
# TIỆN ÍCH ẢNH
# ============================================================

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Chuyển PIL Image -> numpy BGR (OpenCV)."""
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def crop_negative_patches(img_bgr: np.ndarray,
                           n: int = MAX_NEG_CROPS) -> list:
    """
    Cắt ngẫu nhiên n patch WIN_H x WIN_W từ ảnh nền.
    Nếu ảnh nhỏ hơn cửa sổ thì resize và trả về 1 patch.
    """
    h, w    = img_bgr.shape[:2]
    patches = []

    if h < WIN_H or w < WIN_W:
        patches.append(cv2.resize(img_bgr, (WIN_W, WIN_H)))
        return patches

    for _ in range(n):
        y0 = random.randint(0, h - WIN_H)
        x0 = random.randint(0, w - WIN_W)
        patches.append(img_bgr[y0:y0+WIN_H, x0:x0+WIN_W])
    return patches


# ============================================================
# SLIDING WINDOW
# ============================================================

def sliding_window(image: np.ndarray, stride: int = STRIDE):
    """
    Generator: duyệt qua ảnh bằng cửa sổ WIN_H x WIN_W.
    Yield (x0, y0, patch).
    """
    h, w = image.shape[:2]
    for y in range(0, h - WIN_H + 1, stride):
        for x in range(0, w - WIN_W + 1, stride):
            yield x, y, image[y:y+WIN_H, x:x+WIN_W]


# ============================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================

def compute_iou(a: tuple, b: tuple) -> float:
    """Tính IoU giữa 2 box (x0, y0, x1, y1, ...)."""
    ax0, ay0, ax1, ay1 = a[:4]
    bx0, by0, bx1, by1 = b[:4]

    ix0 = max(ax0, bx0);  iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1);  iy1 = min(ay1, by1)

    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list,
        iou_threshold: float = IOU_THRESHOLD) -> list:
    """
    Non-Maximum Suppression.
    Input : list of (x0, y0, x1, y1, confidence)
    Output: list đã lọc, sắp xếp theo confidence giảm dần.
    """
    if not detections:
        return []

    boxes = sorted(detections, key=lambda b: b[4], reverse=True)
    kept  = []

    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes
                 if compute_iou(best, b) < iou_threshold]

    return kept


# ============================================================
# VẼ KẾT QUẢ
# ============================================================

def draw_detections(image_bgr: np.ndarray,
                    detections: list,
                    color: tuple = (0, 255, 0),
                    thickness: int = 2) -> np.ndarray:
    """Vẽ bounding box + confidence lên ảnh, trả về bản sao."""
    out = image_bgr.copy()
    for (x0, y0, x1, y1, conf) in detections:
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
        label = f"Person {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out,
                      (x0, y0 - th - 6), (x0 + tw + 4, y0),
                      color, -1)
        cv2.putText(out, label, (x0 + 2, y0 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def save_result(image_bgr: np.ndarray,
                name: str,
                out_dir: str = OUTPUT_DIR) -> str:
    """Lưu ảnh ra thư mục, trả về đường dẫn file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, image_bgr)
    return path


# ============================================================
# LƯU / LOAD MODEL
# ============================================================

def save_model(knn, scaler, path: str = MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump({"knn": knn, "scaler": scaler}, f)
    print(f"[INFO] Model lưu tại: {path}")


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy model: {path}\n"
            "Hãy chạy train.py trước."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Đã load model: {path}")
    return data["knn"], data["scaler"]
