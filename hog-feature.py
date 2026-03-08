import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure


# ============================================================
# 1. HOG THỦ CÔNG (từ đầu, không dùng thư viện)
# ============================================================

def compute_gradients(image):
    """Tính gradient theo hướng x và y bằng Sobel filter."""
    image = image.astype(np.float64)
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])
    Gx = cv2.filter2D(image, -1, Kx)
    Gy = cv2.filter2D(image, -1, Ky)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    angle     = np.arctan2(Gy, Gx) * (180 / np.pi) % 180  # unsigned [0, 180)
    return magnitude, angle


def build_cell_histogram(magnitude, angle, num_bins=9):
    """
    Xây dựng histogram gradient cho 1 cell.
    num_bins=9  ->  mỗi bin = 20 độ  (0-20, 20-40, …, 160-180)
    """
    bin_size  = 180 / num_bins
    histogram = np.zeros(num_bins)
    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            mag = magnitude[i, j]
            ang = angle[i, j]
            # Nội suy tuyến tính giữa 2 bin liền kề
            bin_idx  = ang / bin_size
            low_bin  = int(np.floor(bin_idx)) % num_bins
            high_bin = (low_bin + 1) % num_bins
            frac     = bin_idx - np.floor(bin_idx)
            histogram[low_bin]  += mag * (1 - frac)
            histogram[high_bin] += mag * frac
    return histogram


def hog_manual(image, cell_size=8, block_size=2, num_bins=9):
    """
    Trích xuất đặc trưng HOG thủ công.

    Tham số
    -------
    image      : ảnh xám (H x W)
    cell_size  : kích thước mỗi cell (pixel)
    block_size : số cell theo mỗi chiều trong 1 block
    num_bins   : số bin histogram

    Trả về
    ------
    feature_vector : vector đặc trưng HOG 1-D
    all_hists      : histogram của từng cell (để trực quan hoá)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    H, W = image.shape
    magnitude, angle = compute_gradients(image)

    # --- Số cell theo mỗi chiều ---
    n_cells_row = H // cell_size
    n_cells_col = W // cell_size

    # --- Histogram cho từng cell ---
    all_hists = np.zeros((n_cells_row, n_cells_col, num_bins))
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            r0, r1 = r * cell_size, (r + 1) * cell_size
            c0, c1 = c * cell_size, (c + 1) * cell_size
            all_hists[r, c] = build_cell_histogram(
                magnitude[r0:r1, c0:c1],
                angle[r0:r1,    c0:c1],
                num_bins
            )

    # --- Chuẩn hoá theo block (L2-norm) ---
    feature_vector = []
    n_blocks_row = n_cells_row - block_size + 1
    n_blocks_col = n_cells_col - block_size + 1
    eps = 1e-6

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = all_hists[r:r+block_size, c:c+block_size, :].flatten()
            block = block / (np.linalg.norm(block) + eps)
            feature_vector.extend(block)

    return np.array(feature_vector), all_hists


# ============================================================
# 2. HOG BẰNG SKIMAGE (nhanh, chính xác)
# ============================================================

def hog_skimage(image, cell_size=8, block_size=2,
                num_bins=9, visualize=True):
    """
    Trích xuất HOG dùng skimage.feature.hog.

    Trả về feature_vector và (tuỳ chọn) ảnh HOG trực quan hoá.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pixels_per_cell = (cell_size, cell_size)
    cells_per_block = (block_size, block_size)

    result = hog(
        image,
        orientations=num_bins,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=visualize,
        feature_vector=True
    )

    if visualize:
        feature_vector, hog_image = result
        hog_image_rescaled = exposure.rescale_intensity(
            hog_image, in_range=(0, 10)
        )
        return feature_vector, hog_image_rescaled
    else:
        return result, None


# ============================================================
# 3. TRỰC QUAN HOÁ
# ============================================================

def visualize_hog(original, hog_img, title="HOG Feature"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Ảnh gốc")
    axes[0].axis('off')

    axes[1].imshow(hog_img, cmap='gray')
    axes[1].set_title(title)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_cell_histograms(all_hists, num_bins=9):
    """Vẽ histogram gradient của 4 cell đầu tiên."""
    bin_centers = np.arange(num_bins) * (180 / num_bins) + (180 / num_bins / 2)
    n_show = min(4, all_hists.shape[0] * all_hists.shape[1])
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3))
    if n_show == 1:
        axes = [axes]

    k = 0
    for r in range(all_hists.shape[0]):
        for c in range(all_hists.shape[1]):
            if k >= n_show:
                break
            axes[k].bar(bin_centers, all_hists[r, c],
                        width=180/num_bins, color='steelblue', edgecolor='k')
            axes[k].set_title(f"Cell ({r},{c})")
            axes[k].set_xlabel("Góc (độ)")
            axes[k].set_ylabel("Biên độ")
            k += 1

    plt.suptitle("Histogram gradient theo từng cell")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. DEMO
# ============================================================

def demo(image_path=None):
    # --- Tạo ảnh mẫu nếu không cung cấp đường dẫn ---
    if image_path is None:
        print("[INFO] Không có ảnh đầu vào -> dùng ảnh mẫu 128x64.")
        image = np.zeros((64, 128), dtype=np.uint8)
        cv2.rectangle(image, (20, 10), (108, 54), 200, -1)
        cv2.circle(image, (64, 32), 20, 100, -1)
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
        image = cv2.resize(image, (128, 64))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # --- Phương pháp 1: HOG thủ công ---
    print("\n=== HOG THỦ CÔNG ===")
    feat_manual, all_hists = hog_manual(gray, cell_size=8, block_size=2, num_bins=9)
    print(f"  Chiều feature vector : {feat_manual.shape[0]}")
    print(f"  Giá trị min/max      : {feat_manual.min():.4f} / {feat_manual.max():.4f}")
    visualize_cell_histograms(all_hists)

    # --- Phương pháp 2: HOG bằng skimage ---
    print("\n=== HOG SKIMAGE ===")
    feat_ski, hog_img = hog_skimage(gray, cell_size=8, block_size=2,
                                     num_bins=9, visualize=True)
    print(f"  Chiều feature vector : {feat_ski.shape[0]}")
    print(f"  Giá trị min/max      : {feat_ski.min():.4f} / {feat_ski.max():.4f}")
    visualize_hog(gray, hog_img, title="HOG (skimage)")

    return feat_manual, feat_ski


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Thay đường dẫn ảnh của bạn vào đây, hoặc để None để dùng ảnh mẫu
    IMAGE_PATH = None  # ví dụ: "person.jpg"

    feat_manual, feat_ski = demo(image_path=IMAGE_PATH)
    print("\nHoàn thành trích xuất đặc trưng HOG.")
