"""
prepare_dataset.py
本機執行：從 Google Drive 下載圖片並整理成 YOLOv8-cls 訓練格式
"""
import os
import shutil
import random
from pathlib import Path
import gdown

# ── 設定 ─────────────────────────────────────────────────────────────────────
FOLDER_ID   = "1p4i4_PUK0FwDXd8C2JW5srdiZxH5uX8m"
RAW_DIR     = Path("raw_data")
DATASET_DIR = Path("dataset")
SPLIT_RATIO = 0.8   # 80% 訓練 / 20% 驗證
SEED        = 42

LABEL_MAP = {
    "1": "dirty",    # 髒
    "3": "turbid",   # 混濁
    "5": "clean",    # 乾淨
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── 下載 ──────────────────────────────────────────────────────────────────────
def download():
    print("📥 從 Google Drive 下載資料集...")
    gdown.download_folder(
        f"https://drive.google.com/drive/folders/{FOLDER_ID}",
        output=str(RAW_DIR),
        quiet=False,
        use_cookies=False,
    )
    print("✅ 下載完成\n")

# ── 整理資料集 ────────────────────────────────────────────────────────────────
def build_dataset():
    random.seed(SEED)

    # 清空舊資料集
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val"]:
        for label in LABEL_MAP.values():
            (DATASET_DIR / split / label).mkdir(parents=True, exist_ok=True)

    for folder_name, label in LABEL_MAP.items():
        src_dir = RAW_DIR / folder_name

        # 若找不到，嘗試模糊比對（例如 "1_dirty" 等）
        if not src_dir.exists():
            matches = [d for d in RAW_DIR.iterdir()
                       if d.is_dir() and folder_name in d.name]
            if matches:
                src_dir = matches[0]
                print(f"⚠️  使用資料夾：{src_dir.name}")
            else:
                print(f"✗ 找不到資料夾 {folder_name}，跳過")
                continue

        images = sorted([f for f in src_dir.rglob("*") if f.suffix.lower() in IMG_EXTS])
        random.shuffle(images)

        cut        = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:cut]
        val_imgs   = images[cut:]

        for img in train_imgs:
            shutil.copy(img, DATASET_DIR / "train" / label / img.name)
        for img in val_imgs:
            shutil.copy(img, DATASET_DIR / "val" / label / img.name)

        print(f"✅ {label:10s}：訓練 {len(train_imgs):3d} 張 / 驗證 {len(val_imgs):3d} 張")

    # 統計
    print("\n📊 資料集統計：")
    for split in ["train", "val"]:
        total = sum(1 for _ in (DATASET_DIR / split).rglob("*") if _.is_file())
        print(f"  {split:5s}：{total} 張")
    print(f"\n📁 資料集路徑：{DATASET_DIR.resolve()}")

# ── 主程式 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not RAW_DIR.exists():
        download()
    else:
        print(f"ℹ️  已有 {RAW_DIR}，跳過下載（刪除該資料夾可重新下載）")
    build_dataset()
