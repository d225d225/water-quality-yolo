# 💧 瓦磘溝水質辨識 — Streamlit 網頁版

把訓練好的 YOLOv8 分類模型 (`best.pt`) 變成一個網頁，
任何人都能上傳照片、即時判斷水質。

## 📁 檔案結構

```
streamlit_app/
├── main.py            # 主程式
├── requirements.txt   # 套件清單
├── best.pt            # ← 你訓練好的模型（自己放進來）
└── README.md          # 本說明
```

## 🚀 部署到 Streamlit Cloud（免費）

1. 把 `best.pt` 複製到 `streamlit_app/` 資料夾
2. `git add . && git commit -m "Add streamlit app" && git push`
3. 到 [share.streamlit.io](https://share.streamlit.io) 用 GitHub 登入
4. 點 **New app**，選這個 repo
5. **Main file path** 填：`streamlit_app/main.py`
6. 按 **Deploy**，等 2–5 分鐘就有網址了

## 💻 本機測試

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run main.py
```

## 📦 模型超過 100MB 怎麼辦？

GitHub 單一檔案上限 100MB。本專案的 `best.pt` 約 10–11 MB，**直接放進 repo 即可**，不用擔心。

但如果你之後換成更大的模型（如 `yolov8m-cls`，約 31MB；或偵測模型可能更大），超過 100MB 時有兩個策略：

| 策略 | 做法 | 適合情境 |
|------|------|---------|
| **雲端下載（推薦）** | 把 `best.pt` 上傳到 GitHub Releases 或 Google Drive，在 `main.py` 的 `MODEL_URL` 填網址，程式啟動時自動下載 | 最簡單，免裝額外工具 |
| **Git LFS** | `git lfs install` → `git lfs track "*.pt"` → 照常 commit | 想把模型版本化管理 |

### 雲端下載做法（最推薦）

1. 到 repo 的 **Releases** → **Create a new release** → 上傳 `best.pt`
2. 複製檔案網址（類似 `https://github.com/帳號/repo/releases/download/v1/best.pt`）
3. 打開 `main.py`，把網址填到最上面的 `MODEL_URL`
4. 把本機的 `best.pt` 從 repo 移除（這樣 repo 就不超標）
5. push → 雲端第一次啟動時會自動下載模型

> 💡 Git LFS 在 Streamlit Cloud 上有時要額外設定，**新手建議用「雲端下載」**最省事。
