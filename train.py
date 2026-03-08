"""
train.py  –  Huấn luyện HOG + KNN trên tập INRIA Person
=========================================================
Dataset : marcelarosalesj/inria-person (Hugging Face)
Output  : knn_hog_model.pkl

Chạy:
  python train.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from utils import (
    K_NEIGHBORS, MAX_NEG_CROPS, RANDOM_SEED,
    MODEL_PATH, OUTPUT_DIR,
    extract_hog, pil_to_cv2,
    crop_negative_patches, save_model
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_inria(split_name: str, label_pos: int = 1) -> tuple:
    """
    Load 1 split từ marcelarosalesj/inria-person.

    - Ảnh dương  (label=1) : resize rồi trích HOG trực tiếp.
    - Ảnh âm     (label=0) : cắt MAX_NEG_CROPS patch ngẫu nhiên
                              rồi trích HOG từng patch.

    Trả về
    ------
    X : np.ndarray (N, D)  float32
    y : np.ndarray (N,)    int32
    """
    ds    = load_dataset("marcelarosalesj/inria-person")
    split = ds[split_name]

    # Xác định tên cột linh hoạt
    col_img = next(
        (c for c in ["image", "img"] if c in split.column_names),
        split.column_names[0]
    )
    col_lbl = next(
        (c for c in ["label", "labels"] if c in split.column_names),
        None
    )

    X, y            = [], []
    pos_cnt, neg_cnt = 0, 0

    for item in split:
        img_bgr = pil_to_cv2(item[col_img])
        label   = int(item[col_lbl]) if col_lbl is not None else -1

        if label == label_pos:                  # ảnh dương
            X.append(extract_hog(img_bgr))
            y.append(1)
            pos_cnt += 1

        elif label == 0:                        # ảnh âm
            for patch in crop_negative_patches(img_bgr):
                X.append(extract_hog(patch))
                y.append(0)
            neg_cnt += 1

    print(f"  [{split_name}]  person={pos_cnt}  "
          f"background≈{neg_cnt * MAX_NEG_CROPS}  "
          f"(từ {neg_cnt} ảnh âm)")

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32))


# ============================================================
# 2. HUẤN LUYỆN KNN
# ============================================================

def train(X_tr: np.ndarray, y_tr: np.ndarray,
          X_te: np.ndarray, y_te: np.ndarray):
    """
    Chuẩn hoá + fit KNN + in báo cáo đánh giá.

    Trả về  knn, scaler
    """
    print(f"\n[INFO] Huấn luyện KNN  (k={K_NEIGHBORS}) ...")
    print(f"  Train : {X_tr.shape[0]} mẫu  |  "
          f"Test  : {X_te.shape[0]} mẫu  |  "
          f"Đặc trưng : {X_tr.shape[1]} chiều")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    knn = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS,
        weights='distance',
        metric='euclidean',
        algorithm='ball_tree',  # nhanh hơn brute force với data lớn
        n_jobs=-1
    )
    knn.fit(X_tr_s, y_tr)

    return knn, scaler, X_te_s, y_te


# ============================================================
# 3. ĐÁNH GIÁ
# ============================================================

def evaluate(knn, X_te_s: np.ndarray, y_te: np.ndarray) -> None:
    """In classification report và lưu confusion matrix."""
    y_pred = knn.predict(X_te_s)

    print("\n[KẾT QUẢ ĐÁNH GIÁ]")
    print(classification_report(
        y_te, y_pred,
        target_names=["Background", "Person"]
    ))

    # Confusion matrix
    cm  = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Background", "Person"]
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix  –  HOG + KNN")
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.show()
    print(f"[INFO] Confusion matrix lưu tại: {cm_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    # Kiểm tra xem model đã tồn tại chưa
    if os.path.exists(MODEL_PATH):
        ans = input(f"Model '{MODEL_PATH}' đã tồn tại. Train lại? [y/N]: ")
        if ans.strip().lower() != 'y':
            print("[INFO] Hủy. Dùng model hiện có.")
            return

    print("[INFO] Đang tải dataset từ Hugging Face ...")
    ds_info = load_dataset(
        "marcelarosalesj/inria-person",
        trust_remote_code=True
    )
    splits = list(ds_info.keys())
    print(f"[INFO] Các split: {splits}")

    train_split = "train" if "train" in splits else splits[0]
    test_split  = "test"  if "test"  in splits else splits[-1]

    # --- Load ---
    print("\n[INFO] Trích xuất đặc trưng HOG ...")
    X_tr, y_tr = load_inria(train_split)
    X_te, y_te = load_inria(test_split)

    # --- Train ---
    knn, scaler, X_te_s, y_te = train(X_tr, y_tr, X_te, y_te)

    # --- Đánh giá ---
    evaluate(knn, X_te_s, y_te)

    # --- Lưu ---
    save_model(knn, scaler, MODEL_PATH)
    print("\n[HOÀN THÀNH] Chạy test.py để nhận diện người trong ảnh.")


if __name__ == "__main__":
    main()
