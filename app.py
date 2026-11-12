"""
Demo final project: Peningkatan Kualitas Citra Cahaya Rendah
Kombinasi CLAHE dan BPDHE pada dataset ExDark.

Jalankan lokal:  streamlit run app.py
"""
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from lowlight import (METHODS, enhance, evaluate, load_calibration,
                      sample_dark_image, PSNR_CAP)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAL = load_calibration(os.path.join(BASE_DIR, "calibration.json"))
MAX_SIDE = 900  # batas sisi terpanjang agar cepat di HP dan laptop

st.set_page_config(page_title="Peningkatan Citra Cahaya Rendah",
                   page_icon="🌙", layout="wide")

st.title("🌙 Peningkatan Kualitas Citra Cahaya Rendah")
st.caption("Kombinasi CLAHE dan BPDHE · Dataset ExDark · Demo final project Pengolahan Citra Digital")

# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Citra masukan")
    source = st.radio("Sumber citra", ["Unggah citra", "Contoh gelap (sintetis)"])
    uploaded = None
    if source == "Unggah citra":
        uploaded = st.file_uploader("Pilih citra (JPG/PNG/BMP)",
                                    type=["jpg", "jpeg", "png", "bmp"])

    st.header("2. Metode")
    method = st.selectbox("Metode", list(METHODS.keys()),
                          format_func=lambda k: METHODS[k],
                          index=list(METHODS).index("kombinasi_adaptif"))

    st.header("3. Parameter")
    clip = st.slider("Clip limit CLAHE", 1.0, 4.0, 2.0, 0.5)
    tile = st.select_slider("Ukuran tile CLAHE", options=[4, 8, 16], value=8)
    sigma = st.slider("Sigma penghalus histogram (BPDHE)", 0.5, 5.0, 2.0, 0.5)
    alpha = st.slider("α bobot tetap (bobot CLAHE)", 0.0, 1.0,
                      float(CAL["alpha_fixed"]), 0.1)
    gamma = st.slider("γ koreksi gamma", 0.4, 0.7, 0.5, 0.05)
    st.caption(f"Koefisien adaptif a = {CAL['a']:.3f}, b = {CAL['b']:.3f} "
               f"(dari calibration.json)")


# ------------------------------------------------------------- helpers
def load_image():
    if source == "Unggah citra":
        if uploaded is None:
            return None
        img = np.array(Image.open(uploaded).convert("RGB"))
    else:
        img = sample_dark_image()
    h, w = img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def value_histogram(img_rgb):
    v = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[..., 2]
    counts = np.bincount(v.ravel(), minlength=256)
    return pd.DataFrame({"jumlah piksel": counts})


def run(img, key, **extra):
    return enhance(img, key, clip_limit=clip, tile=tile, sigma=sigma,
                   alpha=alpha, a=CAL["a"], b=CAL["b"], gamma=gamma, **extra)


def fmt_psnr(v):
    return f"{v:.2f}" + (" (maks)" if v >= PSNR_CAP else "")


# ------------------------------------------------------------- main
img = load_image()
if img is None:
    st.info("Unggah citra terlebih dahulu di panel kiri, atau pilih **Contoh gelap (sintetis)**.")
    st.stop()

out, used_alpha = run(img, method)
metrics = evaluate(img, out)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Sebelum")
    st.image(img, use_container_width=True)
with col_b:
    st.subheader(f"Sesudah · {METHODS[method]}")
    st.image(out, use_container_width=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Entropi (↑)", f"{metrics['Entropi']:.3f}")
m2.metric("SSIM thd. masukan", f"{metrics['SSIM (thd. masukan)']:.3f}")
m3.metric("PSNR thd. masukan (dB)", fmt_psnr(metrics["PSNR dB (thd. masukan)"]))
m4.metric("AMBE (↓)", f"{metrics['AMBE']:.2f}")
m5.metric("Kenaikan kecerahan", f"{metrics['Kenaikan kecerahan']:+.2f}")

if used_alpha is not None:
    label = "adaptif" if method == "kombinasi_adaptif" else "tetap"
    st.info(f"Bobot α {label} untuk citra ini = **{used_alpha:.3f}** "
            f"(α = 1 berarti sepenuhnya CLAHE, α = 0 berarti sepenuhnya BPDHE).")

h1, h2 = st.columns(2)
with h1:
    st.caption("Histogram kanal V · sebelum")
    st.bar_chart(value_histogram(img), height=200)
with h2:
    st.caption("Histogram kanal V · sesudah")
    st.bar_chart(value_histogram(out), height=200)

ok, buf = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
if ok:
    st.download_button("⬇️ Unduh hasil (PNG)", data=buf.tobytes(),
                       file_name="hasil_peningkatan.png", mime="image/png")

with st.expander("📊 Bandingkan semua metode pada citra ini"):
    rows = []
    for key, name in METHODS.items():
        o, _ = run(img, key)
        mm = evaluate(img, o)
        rows.append({
            "Metode": name,
            "Entropi": mm["Entropi"],
            "SSIM": mm["SSIM (thd. masukan)"],
            "PSNR (dB)": mm["PSNR dB (thd. masukan)"],
            "AMBE": mm["AMBE"],
            "Kenaikan kecerahan": mm["Kenaikan kecerahan"],
        })
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)

with st.expander("ℹ️ Cara membaca metrik"):
    st.markdown(
        "- **Entropi**: makin tinggi makin banyak detail yang tampak.\n"
        "- **SSIM dan PSNR** di sini diukur terhadap **citra masukan** (ExDark tidak "
        "memiliki citra acuan terang). Nilai tinggi hanya berarti hasil tidak banyak berubah, "
        "bukan otomatis lebih baik, sehingga dibaca bersama entropi dan kenaikan kecerahan.\n"
        "- **AMBE**: selisih rata-rata kecerahan keluaran dan masukan, makin kecil makin "
        "menjaga kecerahan asli.\n"
        "- **NIQE** (metrik tanpa acuan) dan validasi PSNR/SSIM pada dataset LOL tidak "
        "termasuk dalam demo ini; keduanya dijalankan di skrip eksperimen."
    )
