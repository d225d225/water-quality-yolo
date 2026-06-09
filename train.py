"""
train.py
本機訓練腳本（有 GPU 時使用；無 GPU 請改用 Colab Notebook）
"""
from ultralytics import YOLO
from pathlib import Path

DATASET_DIR = "dataset"
MODEL       = "yolov8s-cls.pt"   # nano=最快, small=平衡, medium=最準
EPOCHS      = 100
IMGSZ       = 224
BATCH       = 16
PROJECT     = "runs/classify"
NAME        = "water_quality"

def main():
    if not Path(DATASET_DIR).exists():
        print("❌ 找不到 dataset 資料夾，請先執行 prepare_dataset.py")
        return

    print(f"🚀 開始訓練 {MODEL}，共 {EPOCHS} epochs")
    model = YOLO(MODEL)

    results = model.train(
        data     = DATASET_DIR,
        epochs   = EPOCHS,
        imgsz    = IMGSZ,
        batch    = BATCH,
        project  = PROJECT,
        name     = NAME,
        patience = 20,        # 20 epoch 無進步提早停止
        exist_ok = True,
    )

    best = Path(PROJECT) / NAME / "weights" / "best.pt"
    print(f"\n🎉 訓練完成！")
    print(f"📦 最佳模型：{best.resolve()}")
    print(f"📊 結果圖表：{Path(PROJECT) / NAME}")

if __name__ == "__main__":
    main()
