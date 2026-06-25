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

---

## 🔬 對照實驗結論：原始照片 vs 裁切水面

本專案（**原始照片**）與姊妹專案 [Image-Cropping-for-Water-Quality-Classification](https://github.com/d225d225/Image-Cropping-for-Water-Quality-Classification)（**裁切水面**）使用同一批瓦磘溝照片、同樣的 YOLOv8s-cls 模型，唯一差別是「要不要先把照片裁切到只剩水面」。

| 項目 | 本專案（原始照片） | 姊妹專案（裁切水面） |
|------|------------------|------------------|
| 輸入 | 完整照片（含背景） | 只留水面 |
| 前處理 | 無，直接訓練 | 每張需手動裁切 |
| 測試準確率 | **約 80%** 🏆 | 58.3% |

### 意外的發現：裁切反而變差

直覺上「去掉天空、河岸等背景雜訊，只留水面」應該讓 AI 更專注、更準。**但實驗結果相反**——裁切版掉到 58.3%，原始版反而有約 80%。

可能原因：
1. **背景其實是有用的線索**：河岸、水草、陽光反光、周遭環境，這些「情境」幫助 AI 判斷水質。裁掉等於丟掉資訊。
2. **裁切後資訊量變少**：只剩水面紋理，乾淨水與混濁水的差異本來就微妙，少了對照物更難分。
3. **裁切引入不一致**：每張裁切範圍、比例不同，反而成為新的雜訊來源。

### 給後續研究者的啟示

> **看似合理的假設，必須用數據驗證，不能只靠直覺。**

「裁切去背景」是一個很自然的想法，但在這個任務上反而有害。這正是科學研究的價值——讓實驗結果說話。本專案（不裁切）為目前較佳方案。
