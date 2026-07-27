# 💧 瓦磘溝水質評估檢測 Web 應用

一個用 **FastAPI + Tailwind CSS** 打造的水質檢測網頁：
上傳照片 → YOLOv8 分類模型判斷 → 顯示水質分數。若不是水的照片會提醒你。

## 📁 目錄結構

```
webapp/
├── main.py              # FastAPI 後端（載入模型 + /predict API）
├── requirements.txt     # Python 依賴
├── best.pt              # ← 你訓練好的模型（自己放進來）
├── static/
│   └── index.html       # 前端介面（拖曳上傳 + 結果顯示）
└── README.md
```

## 🧠 核心邏輯

| 情況 | 判定 | 前端顯示 |
|------|------|---------|
| clean，信心度 ≥ 60% | 5 分 | ✅ 乾淨 |
| turbid，信心度 ≥ 60% | 3 分 | ⚠️ 混濁 |
| dirty，信心度 ≥ 60% | 1 分 | ❌ 髒 |
| 信心度 < 60% | 非水體 | 🤨「別開玩笑！這看起來不是水的照片。」 |

> **非水體偵測原理：** 模型只認得三種水質，任何圖都會被硬分類。
> 但若最高信心度低於門檻（模型自己也很猶豫），就判定「這不是水」。
> 門檻可在 `main.py` 的 `CONF_THRESHOLD` 調整（預設 0.60）。

## 🚀 啟動步驟

```bash
# 1. 進入資料夾
cd webapp

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 把訓練好的 best.pt 放進 webapp/ 資料夾

# 4. 啟動伺服器
uvicorn main:app --reload --port 8000
```

打開瀏覽器 → http://localhost:8000

## 🧪 測試

- **正常測試**：上傳一張瓦磘溝水面照片，應顯示分數與各類別機率。
- **例外測試**：上傳一張人臉或風景照，應跳出「別開玩笑！這看起來不是水的照片。」
- **API 測試**（用 curl）：
  ```bash
  curl -X POST http://localhost:8000/predict -F "file=@test.jpg"
  ```
- **健康檢查**：http://localhost:8000/health → 確認 `model_found: true`

## ⚠️ 常見問題

| 問題 | 解法 |
|------|------|
| 啟動報「找不到 best.pt」 | 把模型檔放到 `webapp/` 資料夾 |
| ultralytics 安裝報錯 | 確認 `numpy < 2.0`（requirements 已鎖定） |
| 全部照片都判成「非水體」 | 門檻太高，調低 `CONF_THRESHOLD` |
| 什麼都判成有水（誤判） | 門檻太低，調高 `CONF_THRESHOLD` |

## 📦 部署建議

- **本機 / 校內電腦**：照上面步驟用 uvicorn 跑即可。
- **雲端**：可部署到 Render、Railway、或有 GPU 的主機。
  模型約 10MB，直接放進 repo 即可（未超過 GitHub 100MB 限制）。
