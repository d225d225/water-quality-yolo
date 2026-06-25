# -*- coding: utf-8 -*-
"""
瓦磘溝水質辨識 — Streamlit 網頁版
========================================
使用者上傳水域照片 → YOLOv8 分類模型判斷水質等級 → 顯示分數與信心度。

這份程式給高中師生使用，註解力求淺顯易懂。
"""

import streamlit as st
from PIL import Image
from pathlib import Path
import os

# ── ① 載入 AI 套件 ──────────────────────────────────────────────
# ultralytics 是 YOLOv8 的官方套件，YOLO 類別可以載入我們訓練好的模型
from ultralytics import YOLO


# ── ② 設定區 ────────────────────────────────────────────────────
# 模型檔放在跟這支程式同一個資料夾，檔名 best.pt
MODEL_PATH = Path(__file__).parent / "best.pt"

# 如果模型太大（>100MB）放在雲端，這裡填網址，程式會自動下載（見下方說明）
MODEL_URL = ""   # 例如 "https://github.com/帳號/repo/releases/download/v1/best.pt"

# 水質分數對應表：用「類別名稱」對應，不靠數字編號，比較保險
#   clean（乾淨）  -> 5 分
#   turbid（混濁） -> 3 分
#   dirty（髒）    -> 1 分
SCORE_MAP = {
    "clean":  {"score": 5, "label": "5 分（乾淨）", "emoji": "✅", "color": "#2e7d32"},
    "turbid": {"score": 3, "label": "3 分（混濁）", "emoji": "⚠️", "color": "#f9a825"},
    "dirty":  {"score": 1, "label": "1 分（髒）",   "emoji": "❌", "color": "#c62828"},
}


# ── ③ 載入模型（含快取）─────────────────────────────────────────
@st.cache_resource   # ← 關鍵！讓模型只載入「一次」，之後重複使用，省記憶體又變快
def load_model():
    """載入 YOLOv8 分類模型。若本機沒有檔案，且有設定網址，就自動下載。"""
    # 情況一：模型在雲端，本機沒有 → 自動下載
    if not MODEL_PATH.exists() and MODEL_URL:
        import urllib.request
        with st.spinner("正在下載 AI 模型，請稍候…"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    # 情況二：還是找不到模型 → 給清楚的錯誤訊息
    if not MODEL_PATH.exists():
        st.error(
            "找不到模型檔 best.pt。\n\n"
            "請把訓練好的 best.pt 放在與 main.py 相同的資料夾，"
            "或在程式上方的 MODEL_URL 填入下載網址。"
        )
        st.stop()

    return YOLO(str(MODEL_PATH))


# ── ④ 推論：判斷一張照片的水質 ──────────────────────────────────
def predict_water_quality(model, pil_image):
    """
    輸入一張照片，回傳：
      - 最可能的類別名稱（如 clean）
      - 信心度（0~1）
      - 全部類別的機率（給進階顯示用）
    """
    # YOLOv8 分類預測
    results = model.predict(pil_image, verbose=False)
    r = results[0]

    # results[0].probs 裡有分類機率
    probs      = r.probs                      # 機率物件
    top1_idx   = int(probs.top1)              # 信心最高的類別編號
    top1_conf  = float(probs.top1conf)        # 該類別的信心度
    class_name = model.names[top1_idx]        # 編號轉成名稱（clean/turbid/dirty）

    # 整理每個類別的機率，方便畫長條圖
    all_probs = {model.names[i]: float(p) for i, p in enumerate(probs.data.tolist())}

    return class_name, top1_conf, all_probs


# ── ⑤ 網頁介面 ──────────────────────────────────────────────────
st.set_page_config(page_title="瓦磘溝水質辨識", page_icon="💧", layout="centered")

st.title("💧 瓦磘溝水質辨識系統")
st.caption("瓦磘溝行動研究社 × YOLOv8 影像分類")

# 運作原理說明（可展開）
with st.expander("📖 運作原理說明（點我展開）", expanded=False):
    st.markdown(
        """
        這個系統用 **YOLOv8 影像分類模型** 來判斷水質。

        1. 我們蒐集了瓦磘溝的水面照片，分成三類：**乾淨 / 混濁 / 髒**。
        2. 用這些照片「訓練」AI，讓它學會分辨不同水質的樣子。
        3. 你上傳一張新照片，AI 就會告訴你它「最像」哪一類，
           並給出**信心度**（它有多確定）。

        > ⚠️ 這是 AI 的預測結果，僅供教學與初步參考，
        > 正式的水質判定仍需專業儀器檢測。
        """
    )

# 側邊欄：信心度門檻滑桿
st.sidebar.header("⚙️ 設定")
conf_threshold = st.sidebar.slider(
    "信心度門檻（Confidence Threshold）",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="低於這個信心度的預測會被標示為『不確定』，避免亂猜。",
)

# 載入模型
model = load_model()

# 圖片上傳區
uploaded = st.file_uploader(
    "📤 請上傳一張水域照片（jpg / png）",
    type=["jpg", "jpeg", "png"],
)

# ── ⑥ 主流程：有上傳才預測 ─────────────────────────────────────
if uploaded is None:
    # 錯誤處理：使用者還沒上傳圖片
    st.info("👆 請先上傳一張照片，AI 才能開始判斷喔！")
else:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("這張圖片讀取失敗，請換一張 jpg 或 png 試試。")
        st.stop()

    # 顯示上傳的照片
    st.image(image, caption="你上傳的照片", use_column_width=True)

    # 執行預測
    with st.spinner("AI 判斷中…"):
        class_name, conf, all_probs = predict_water_quality(model, image)

    st.divider()
    st.subheader("🔍 判斷結果")

    info = SCORE_MAP.get(class_name)

    if info is None:
        # 防呆：萬一模型類別名稱跟預期不同
        st.warning(f"模型回傳了未知類別「{class_name}」，請確認模型標籤設定。")
    elif conf < conf_threshold:
        # 信心度太低 → 不直接下結論
        st.warning(
            f"⚠️ AI 不太確定（信心度只有 {conf:.1%}，低於門檻 {conf_threshold:.0%}）。\n\n"
            f"最接近的判斷是：{info['label']}，但建議換一張更清楚的照片。"
        )
    else:
        # 正常顯示結果
        st.markdown(
            f"<h2 style='color:{info['color']}'>{info['emoji']} {info['label']}</h2>",
            unsafe_allow_html=True,
        )
        st.metric(label="最高信心度（Top-1 Confidence）", value=f"{conf:.1%}")

    # 顯示三個類別的機率長條圖
    st.divider()
    st.caption("各類別機率分布")
    chart_data = {
        SCORE_MAP[name]["label"] if name in SCORE_MAP else name: prob
        for name, prob in all_probs.items()
    }
    st.bar_chart(chart_data)

# 頁尾
st.divider()
st.caption("製作：瓦磘溝行動研究社　|　模型：YOLOv8s-cls　|　僅供教學參考")
