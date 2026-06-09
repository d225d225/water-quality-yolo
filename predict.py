"""
predict.py
使用訓練好的模型辨識水質
用法：python predict.py <圖片路徑或資料夾>
"""
import sys
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/classify/water_quality/weights/best.pt"

LABEL_ZH = {
    "clean":  "✅ 乾淨",
    "turbid": "⚠️  混濁",
    "dirty":  "❌ 髒",
}

def predict(source: str):
    if not Path(MODEL_PATH).exists():
        print(f"❌ 找不到模型：{MODEL_PATH}")
        print("請先執行 train.py 或從 GitHub Releases 下載 best.pt")
        return

    model   = YOLO(MODEL_PATH)
    results = model(source)

    for r in results:
        probs = r.probs
        top1  = r.names[probs.top1]
        conf  = probs.top1conf.item()

        print(f"\n{'─'*40}")
        print(f"圖片：{r.path}")
        print(f"預測：{LABEL_ZH.get(top1, top1)}  （信心度 {conf:.1%}）")
        print("各類別機率：")
        for i, name in r.names.items():
            bar = "█" * int(probs.data[i].item() * 20)
            print(f"  {LABEL_ZH.get(name, name):14s} {probs.data[i].item():5.1%}  {bar}")

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "test_photo.jpg"
    predict(source)
