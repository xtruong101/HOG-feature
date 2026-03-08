"""
Nhận diện người trong ảnh
=========================
Pipeline:
  1. Load tập INRIA Person (marcelarosalesj/inria-person) từ Hugging Face
  2. Trích xuất đặc trưng HOG cho từng mẫu
  3. Huấn luyện KNN classifier
  4. Sliding Window + Scale Pyramid trên ảnh test
  5. Non-Maximum Suppression (NMS) lọc bounding box
  6. Lưu/hiển thị ảnh kết quả

Thư viện cần:
  pip install numpy opencv-python scikit-image scikit-learn datasets pillow matplotlib
"""

import os
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from PIL import Image
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from datasets import load_dataset

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================

WIN_H         = 128     # chiều cao cửa sổ HOG (pixel)
WIN_W         = 64      # chiều rộng cửa sổ HOG (pixel)
CELL_SIZE     = 8       # kích thước cell HOG (pixel)
BLOCK_SIZE    = 2       # số cell / chiều trong 1 block
NUM_BINS      = 9       # số bin histogram
STRIDE        = 16      # bước trượt cửa sổ (pixel)
SCALES        = [1.0, 0.85, 0.7, 0.55, 0.4]  # tỉ lệ scale pyramid
K_NEIGHBORS   = 7       # k trong KNN
IOU_THRESHOLD = 0.3     # ngưỡng IoU cho NMS
CONF_THRESHOLD= 0.65    # ngưỡng xác suất để coi là "người"
MAX_NEG_CROPS = 15      # số patch cắt từ mỗi ảnh âm
RANDOM_SEED   = 42
MODEL_PATH    = "knn_hog_model.pkl"   # file lưu model
OUTPUT_DIR    = "detection_results"   # thư mục lưu kết quả

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 1. HOG FEATURE EXTRACTION
# ============================================================

def extract_hog(image: np.ndarray) -> np.ndarray:
    """
    Trích xuất HOG từ ảnh đầu vào.

    Tham số
    -------
    image : ảnh numpy (bất kỳ số kênh, bất kỳ kích thước)

    Trả về
    ------
    feature : vector 1-D numpy float32
    """
    # Chuyển sang xám
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Resize về kích thước chuẩn
    gray = cv2.resize(gray, (WIN_W, WIN_H))

    feature = hog(
        gray,
        orientations=NUM_BINS,
        pixels_per_cell=(CELL_SIZE, CELL_SIZE),
        cells_per_block=(BLOCK_SIZE, BLOCK_SIZE),
        block_norm='L2-Hys',
        feature_vector=True
    )
    return feature.astype(np.float32)


# ============================================================
# 2. LOAD DATASET TỪ HUGGING FACE
# ============================================================

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Chuyển PIL Image -> numpy BGR (OpenCV)."""
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def crop_negative_patches(img_bgr: np.ndarray, n: int = MAX_NEG_CROPS) -> list:
    """
    Cắt ngẫu nhiên n patch WIN_H x WIN_W từ ảnh nền lớn.
    Nếu ảnh nhỏ hơn cửa sổ thì resize và trả về 1 patch.
    """
    h, w = img_bgr.shape[:2]
    patches = []

    if h < WIN_H or w < WIN_W:
        patches.append(cv2.resize(img_bgr, (WIN_W, WIN_H)))
        return patches

    for _ in range(n):
        y0 = random.randint(0, h - WIN_H)
        x0 = random.randint(0, w - WIN_W)
        patches.append(img_bgr[y0:y0+WIN_H, x0:x0+WIN_W])
    return patches


def load_inria_dataset(split_train="train", split_test="test",
                       max_pos=None, max_neg=None):
    """
    Load tập marcelarosalesj/inria-person từ Hugging Face.

    Trả về
    ------
    X_train, y_train : feature train
    X_test,  y_test  : feature test
    test_images_bgr  : danh sách ảnh gốc test (để chạy detection)
    """
    print("[INFO] Đang tải dataset từ Hugging Face ...")
    ds = load_dataset("marcelarosalesj/inria-person")
    print(f"[INFO] Các split có sẵn: {list(ds.keys())}")

    def process_split(split_name, max_p=None, max_n=None):
        split = ds[split_name]
        X, y = [], []

        # --- Xác định tên cột ---
        col_img   = "image"   if "image"   in split.column_names else split.column_names[0]
        col_label = "label"   if "label"   in split.column_names else \
                    "labels"  if "labels"  in split.column_names else None

        pos_count, neg_count = 0, 0

        for item in split:
            pil_img = item[col_img]
            label   = int(item[col_label]) if col_label else -1
            img_bgr = pil_to_cv2(pil_img)

            if label == 1:                          # ảnh dương (có người)
                if max_p and pos_count >= max_p:
                    continue
                feat = extract_hog(img_bgr)
                X.append(feat)
                y.append(1)
                pos_count += 1

            elif label == 0:                        # ảnh âm (nền)
                if max_n and neg_count >= max_n:
                    continue
                patches = crop_negative_patches(img_bgr)
                for patch in patches:
                    feat = extract_hog(patch)
                    X.append(feat)
                    y.append(0)
                neg_count += 1

        print(f"  [{split_name}] person={pos_count}  background≈{neg_count*MAX_NEG_CROPS}")
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    # --- Lấy ảnh test gốc để chạy detection ---
    test_images_bgr = []
    test_split_name = split_test if split_test in ds else split_train

    for item in ds[test_split_name]:
        col_img = "image" if "image" in ds[test_split_name].column_names \
                           else ds[test_split_name].column_names[0]
        col_lbl = "label" if "label" in ds[test_split_name].column_names else None
        if col_lbl:
            if int(item[col_lbl]) == 1:
                test_images_bgr.append(pil_to_cv2(item[col_img]))
                if len(test_images_bgr) >= 10:
                    break

    X_train, y_train = process_split(split_train, max_pos, max_neg)
    X_test,  y_test  = process_split(
        split_test if split_test in ds else split_train
    )

    return X_train, y_train, X_test, y_test, test_images_bgr


# ============================================================
# 3. HUẤN LUYỆN KNN
# ============================================================

def train_knn(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray,  y_test: np.ndarray):
    """
    Chuẩn hoá + huấn luyện KNN.

    Trả về
    ------
    knn    : KNeighborsClassifier đã fit
    scaler : StandardScaler đã fit
    """
    print(f"\n[INFO] Huấn luyện KNN (k={K_NEIGHBORS}) ...")
    print(f"  Train: {X_train.shape[0]} mẫu  |  Test: {X_test.shape[0]} mẫu")

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)

    knn = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS,
        weights='distance',     # láng giềng gần hơn có trọng số cao hơn
        metric='euclidean',
        n_jobs=-1
    )
    knn.fit(X_tr, y_train)

    # --- Đánh giá ---
    y_pred = knn.predict(X_te)
    print("\n[KẾT QUẢ ĐÁNH GIÁ]")
    print(classification_report(y_test, y_pred,
                                 target_names=["Background", "Person"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  {'':16s} Pred-Neg  Pred-Pos")
    print(f"  {'True-Neg':16s}  {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"  {'True-Pos':16s}  {cm[1,0]:6d}    {cm[1,1]:6d}")

    return knn, scaler


def save_model(knn, scaler, path=MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump({"knn": knn, "scaler": scaler}, f)
    print(f"[INFO] Model lưu tại: {path}")


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Đã load model từ: {path}")
    return data["knn"], data["scaler"]


# ============================================================
# 4. SLIDING WINDOW + SCALE PYRAMID
# ============================================================

def sliding_window(image: np.ndarray, stride: int = STRIDE):
    """
    Generator: trả về (x0, y0, patch) cho cửa sổ WIN_H x WIN_W.
    """
    h, w = image.shape[:2]
    for y in range(0, h - WIN_H + 1, stride):
        for x in range(0, w - WIN_W + 1, stride):
            yield x, y, image[y:y+WIN_H, x:x+WIN_W]


def detect_persons(image_bgr: np.ndarray,
                   knn: KNeighborsClassifier,
                   scaler: StandardScaler,
                   conf_threshold: float = CONF_THRESHOLD,
                   stride: int = STRIDE,
                   scales: list = None) -> list:
    """
    Chạy phát hiện người trên 1 ảnh (multi-scale sliding window).

    Trả về
    ------
    detections : list of (x0, y0, x1, y1, confidence)
                 tọa độ trên ảnh gốc
    """
    if scales is None:
        scales = SCALES

    h0, w0 = image_bgr.shape[:2]
    detections = []

    for scale in scales:
        new_h = max(WIN_H, int(h0 * scale))
        new_w = max(WIN_W, int(w0 * scale))
        resized = cv2.resize(image_bgr, (new_w, new_h))

        for x, y, patch in sliding_window(resized, stride):
            feat   = extract_hog(patch).reshape(1, -1)
            feat_s = scaler.transform(feat)
            proba  = knn.predict_proba(feat_s)[0]   # [P(bg), P(person)]

            if len(proba) < 2:
                continue
            conf = proba[1]                          # xác suất là người

            if conf >= conf_threshold:
                # Ánh xạ tọa độ về ảnh gốc
                x0 = int(x  / scale)
                y0 = int(y  / scale)
                x1 = int((x + WIN_W) / scale)
                y1 = int((y + WIN_H) / scale)
                detections.append((x0, y0, x1, y1, conf))

    return detections


# ============================================================
# 5. NON-MAXIMUM SUPPRESSION (NMS)
# ============================================================

def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Tính IoU giữa 2 bounding box (x0,y0,x1,y1,...)."""
    ax0, ay0, ax1, ay1 = box_a[:4]
    bx0, by0, bx1, by1 = box_b[:4]

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter   = inter_w * inter_h

    area_a  = (ax1 - ax0) * (ay1 - ay0)
    area_b  = (bx1 - bx0) * (by1 - by0)
    union   = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def nms(detections: list, iou_threshold: float = IOU_THRESHOLD) -> list:
    """
    Non-Maximum Suppression.

    Tham số
    -------
    detections    : list of (x0, y0, x1, y1, confidence)
    iou_threshold : IoU > threshold -> giữ box có điểm cao hơn, bỏ box kia

    Trả về
    ------
    kept : list các box sau NMS
    """
    if not detections:
        return []

    # Sắp xếp giảm dần theo confidence
    boxes = sorted(detections, key=lambda b: b[4], reverse=True)
    kept  = []

    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if compute_iou(best, b) < iou_threshold]

    return kept


# ============================================================
# 6. VISUALIZE & LƯU KẾT QUẢ
# ============================================================

def draw_detections(image_bgr: np.ndarray,
                    detections: list,
                    color: tuple = (0, 255, 0),
                    thickness: int = 2) -> np.ndarray:
    """Vẽ bounding box + confidence score lên ảnh, trả về ảnh đã vẽ."""
    out = image_bgr.copy()
    for (x0, y0, x1, y1, conf) in detections:
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
        label = f"Person {conf:.2f}"
        # Nền chữ
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x0, y0 - th - 6), (x0 + tw + 4, y0), color, -1)
        cv2.putText(out, label, (x0 + 2, y0 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def save_result(image_bgr: np.ndarray, name: str, out_dir: str = OUTPUT_DIR):
    """Lưu ảnh kết quả ra thư mục output."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, image_bgr)
    print(f"  [Lưu] {path}")
    return path


def show_results_grid(result_images: list, titles: list):
    """Hiển thị lưới ảnh kết quả."""
    n = len(result_images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for i, (img_bgr, title) in enumerate(zip(result_images, titles)):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    legend = mpatches.Patch(color='lime', label='Detected Person')
    fig.legend(handles=[legend], loc='lower center', ncol=1, fontsize=11)
    plt.suptitle("Kết quả nhận diện người (HOG + KNN + Sliding Window)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "detection_grid.png"), dpi=120)
    plt.show()
    print(f"[INFO] Đã lưu lưới kết quả: {OUTPUT_DIR}/detection_grid.png")


# ============================================================
# 7. PIPELINE CHÍNH
# ============================================================

def run_detection_on_images(image_list: list,
                             knn: KNeighborsClassifier,
                             scaler: StandardScaler) -> list:
    """
    Chạy toàn bộ pipeline phát hiện người trên danh sách ảnh.

    Trả về
    ------
    result_images : list ảnh BGR đã vẽ bounding box
    """
    result_images = []
    for idx, img_bgr in enumerate(image_list):
        print(f"  [Ảnh {idx+1}/{len(image_list)}] kích thước {img_bgr.shape[1]}x{img_bgr.shape[0]} ...",
              end=" ")

        raw_dets = detect_persons(img_bgr, knn, scaler)
        final    = nms(raw_dets)

        print(f"  raw={len(raw_dets)}  sau NMS={len(final)}")

        out_img = draw_detections(img_bgr, final)
        result_images.append(out_img)

        save_result(out_img, f"result_{idx+1:03d}.jpg")

    return result_images


def main():
    # ----------------------------------------------------------
    # BƯỚC 1: Load dataset
    # ----------------------------------------------------------
    retrain = not os.path.exists(MODEL_PATH)

    if retrain:
        X_train, y_train, X_test, y_test, test_imgs = load_inria_dataset()

        # ----------------------------------------------------------
        # BƯỚC 2 + 3: Trích xuất HOG và huấn luyện KNN
        # ----------------------------------------------------------
        knn, scaler = train_knn(X_train, y_train, X_test, y_test)
        save_model(knn, scaler)
    else:
        print(f"[INFO] Tìm thấy model đã huấn luyện: {MODEL_PATH}")
        print("[INFO] Chỉ load ảnh test từ dataset ...")
        knn, scaler = load_model()

        ds = load_dataset("marcelarosalesj/inria-person")
        split_name = "test" if "test" in ds else "train"
        test_imgs = []
        for item in ds[split_name]:
            col_img = "image" if "image" in ds[split_name].column_names \
                               else ds[split_name].column_names[0]
            col_lbl = "label" if "label" in ds[split_name].column_names else None
            if col_lbl and int(item[col_lbl]) == 1:
                test_imgs.append(pil_to_cv2(item[col_img]))
                if len(test_imgs) >= 10:
                    break

    if not test_imgs:
        print("[WARN] Không lấy được ảnh test từ dataset, dùng ảnh mẫu.")
        sample = np.random.randint(0, 200, (400, 300, 3), dtype=np.uint8)
        test_imgs = [sample]

    # ----------------------------------------------------------
    # BƯỚC 4 + 5: Sliding Window + NMS trên ảnh test
    # ----------------------------------------------------------
    print(f"\n[INFO] Chạy nhận diện trên {len(test_imgs)} ảnh test ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result_images = run_detection_on_images(test_imgs, knn, scaler)

    # ----------------------------------------------------------
    # BƯỚC 6: Hiển thị kết quả
    # ----------------------------------------------------------
    titles = [f"Ảnh test {i+1}" for i in range(len(result_images))]
    show_results_grid(result_images, titles)

    print(f"\n[HOÀN THÀNH] Kết quả lưu tại thư mục: ./{OUTPUT_DIR}/")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
