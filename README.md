# 瓦磘溝水質辨識 — YOLOv8 影像分類

利用瓦磘溝行動研究社學生拍攝的照片，訓練 YOLOv8 模型自動辨識水質等級。

## 水質等級

| 標籤 | 說明 | 資料夾 |
|------|------|--------|
| `clean` ✅ | 乾淨 | `5/` |
| `turbid` ⚠️ | 混濁 | `3/` |
| `dirty` ❌ | 髒 | `1/` |

## 快速開始（Google Colab）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d225d225/water-quality-yolo/blob/main/water_quality_colab.ipynb)

點擊上方按鈕，直接在 Colab 執行訓練與推論。

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `water_quality_colab.ipynb` | 完整 Colab Notebook（訓練 + 推論） |
| `prepare_dataset.py` | 本機資料集整理腳本 |
| `train.py` | 本機訓練腳本 |
| `predict.py` | 推論腳本 |
| `requirements.txt` | Python 套件需求 |

## 資料來源

Google Drive：
https://drive.google.com/drive/folders/1p4i4_PUK0FwDXd8C2JW5srdiZxH5uX8m

## 技術架構

| 項目 | 內容 |
|------|------|
| 模型 | YOLOv8s-cls（Ultralytics） |
| 輸入尺寸 | 224 × 224 |
| 訓練平台 | Google Colab（免費 GPU） |
| 資料儲存 | Google Drive |
| 模型儲存 | GitHub Releases |
