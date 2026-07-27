# -*- coding: utf-8 -*-
"""
瓦磘溝水質評估檢測 — Web 應用後端 (FastAPI)
=================================================
- 載入訓練好的 YOLOv8 分類模型 (best.pt)
- 提供 /predict API：接收圖片 → 回傳水質分數與信心度
- 非水體圖片偵測：信心度過低時回傳 not_water 特殊標記

給高中師生使用，註解力求淺顯易懂。
"""

from pathlib import Path
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from ultralytics import YOLO

# ── 基本設定 ─────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best.pt"          # 訓練好的模型放這裡
STATIC_DIR = BASE_DIR / "static"

# 信心度門檻：最高信心度低於此值 → 判定為「不是水的照片」
CONF_THRESHOLD = 0.60

# 水質分數對應表（用「類別名稱」對應，不靠數字編號，避免標籤順序錯亂）
SCORE_MAP = {
    "clean":  {"score": 5, "label_zh": "乾淨", "label_en": "Clean / Clear Water",      "emoji": "✅"},
    "turbid": {"score": 3, "label_zh": "混濁", "label_en": "Turbid / Moderate Pollution", "emoji": "⚠️"},
    "dirty":  {"score": 1, "label_zh": "髒",   "label_en": "Dirty / Heavy Pollution",   "emoji": "❌"},
}

# ── 載入模型（啟動時只載入一次）──────────────────────────────
app = FastAPI(title="瓦磘溝水質檢測 API")

_model = None   # 全域快取，避免每次請求都重載

def get_model():
    """延遲載入模型；若檔案不存在給清楚錯誤。"""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="找不到模型檔 best.pt，請放到 webapp/ 資料夾。",
            )
        _model = YOLO(str(MODEL_PATH))
    return _model


# ── 推論 API ─────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    接收一張圖片，回傳水質判定結果。
    回傳格式（正常）:
      { "ok": True, "is_water": True, "score": 5, "label_zh": "乾淨",
        "label_en": "Clean / Clear Water", "emoji": "✅",
        "confidence": 0.87, "probs": {"clean":0.87,...} }
    回傳格式（非水體）:
      { "ok": True, "is_water": False, "message": "別開玩笑！這看起來不是水的照片。",
        "confidence": 0.41 }
    """
    # 1) 基本檢查：一定要是圖片
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="請上傳圖片檔（jpg / png）。")

    # 2) 讀取圖片
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="圖片讀取失敗，請換一張試試。")

    # 3) YOLOv8 分類推論
    model   = get_model()
    results = model.predict(image, verbose=False)
    r       = results[0]

    top1_idx   = int(r.probs.top1)
    confidence = float(r.probs.top1conf)
    class_name = model.names[top1_idx]
    all_probs  = {model.names[i]: round(float(p), 4)
                  for i, p in enumerate(r.probs.data.tolist())}

    # 4) 非水體判定：信心度過低 → 不是水
    if confidence < CONF_THRESHOLD:
        return JSONResponse({
            "ok": True,
            "is_water": False,
            "message": "別開玩笑！這看起來不是水的照片。",
            "confidence": round(confidence, 4),
        })

    # 5) 防呆：類別名稱不在對應表
    info = SCORE_MAP.get(class_name)
    if info is None:
        return JSONResponse({
            "ok": True,
            "is_water": False,
            "message": "別開玩笑！這看起來不是水的照片。",
            "confidence": round(confidence, 4),
        })

    # 6) 正常回傳
    return JSONResponse({
        "ok": True,
        "is_water": True,
        "class_name": class_name,
        "score": info["score"],
        "label_zh": info["label_zh"],
        "label_en": info["label_en"],
        "emoji": info["emoji"],
        "confidence": round(confidence, 4),
        "probs": all_probs,
    })


@app.get("/health")
def health():
    """健康檢查：確認模型能載入。"""
    ok = MODEL_PATH.exists()
    return {"ok": ok, "model_found": ok, "threshold": CONF_THRESHOLD}


# ── 提供前端網頁 ─────────────────────────────────────────────
# 首頁回傳 index.html
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

# 其餘靜態檔（若日後有 css/js 檔）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
