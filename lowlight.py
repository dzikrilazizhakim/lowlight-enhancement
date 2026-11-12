"""
Peningkatan citra cahaya rendah: HE, CLAHE, BPDHE, kombinasi CLAHE + BPDHE
(bobot tetap dan adaptif), koreksi gamma, serta metrik evaluasi.

Semua operasi bekerja pada kanal V (ruang warna HSV) agar warna citra asli
tidak banyak berubah, sesuai Bagian 5 pada naskah Tugas 2.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.metrics import structural_similarity

METHODS = {
    "asli": "Citra asli (S0)",
    "he": "Histogram Equalization global (S1)",
    "clahe": "CLAHE (S2)",
    "bpdhe": "BPDHE (S3)",
    "kombinasi_tetap": "Kombinasi CLAHE + BPDHE, bobot tetap (S4)",
    "kombinasi_adaptif": "Kombinasi CLAHE + BPDHE, bobot adaptif (S5)",
    "gamma": "Koreksi gamma (S6)",
}

PSNR_CAP = 99.0  # dipakai bila citra identik (MSE = 0)

DEFAULT_CALIBRATION = {
    "alpha_fixed": 0.5,  # bobot tetap, nilai awal sebelum kalibrasi
    "a": 0.5,            # koefisien adaptif (Persamaan 3), nilai netral
    "b": 0.0,
    "catatan": "Nilai default belum dikalibrasi. Jalankan calibrate.py pada set tuning.",
}


# ---------------------------------------------------------------- utilitas
def ensure_rgb_uint8(img) -> np.ndarray:
    """Pastikan citra berbentuk HxWx3 dan bertipe uint8 (RGB)."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def load_calibration(path: str) -> dict:
    """Baca calibration.json jika ada, jika tidak pakai nilai default."""
    cal = dict(DEFAULT_CALIBRATION)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cal.update(json.load(f))
    return cal


# ---------------------------------------------------------- algoritma inti
def he_channel(v: np.ndarray) -> np.ndarray:
    """Histogram equalization global (Persamaan 1)."""
    return cv2.equalizeHist(v)


def clahe_channel(v: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE dengan clip limit dan ukuran tile (Bagian 5.2)."""
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile), int(tile)))
    return clahe.apply(v)


def bpdhe_channel(v: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    BPDHE (Ibrahim & Kong, 2007) dengan langkah sesuai Bagian 5.3:
    1) histogram dihaluskan Gaussian, 2) titik maksimum lokal dicari,
    3) histogram dipecah pada titik tsb, dan rentang keluaran tiap bagian
    sebanding dengan lebarnya, 4) tiap bagian di-equalize, 5) rata-rata
    keluaran disamakan dengan rata-rata masukan.
    """
    v = v.astype(np.uint8)
    hist = np.bincount(v.ravel(), minlength=256).astype(np.float64)
    hs = gaussian_filter1d(hist, sigma=float(sigma), mode="nearest")

    peaks = [i for i in range(1, 255)
             if hs[i] > 0 and hs[i] > hs[i - 1] and hs[i] >= hs[i + 1]]
    edges = sorted(set([0] + peaks + [255]))
    total_width = float(edges[-1] - edges[0])

    lut = np.zeros(256, dtype=np.float64)
    out_start = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        out_len = 255.0 * (hi - lo) / total_width
        pdf = hs[lo:hi + 1]
        s = pdf.sum()
        if s > 0:
            cdf = np.cumsum(pdf) / s
        else:
            cdf = np.linspace(1.0 / len(pdf), 1.0, len(pdf))
        lut[lo:hi + 1] = out_start + out_len * cdf
        out_start += out_len

    mapped = lut[v]
    target = float(v.mean())

    # Cari pergeseran s sehingga rata-rata setelah pemotongan ke [0, 255]
    # sama dengan rata-rata masukan (bisection; fungsinya monoton naik).
    lo_s, hi_s = -255.0, 255.0
    for _ in range(50):
        mid = 0.5 * (lo_s + hi_s)
        if np.clip(mapped + mid, 0, 255).mean() < target:
            lo_s = mid
        else:
            hi_s = mid
    shift = 0.5 * (lo_s + hi_s)
    return np.clip(np.round(mapped + shift), 0, 255).astype(np.uint8)


def blend_channel(v1: np.ndarray, v2: np.ndarray, alpha: float) -> np.ndarray:
    """V' = alpha * V1 + (1 - alpha) * V2 (Persamaan 2)."""
    a = float(np.clip(alpha, 0.0, 1.0))
    out = a * v1.astype(np.float64) + (1.0 - a) * v2.astype(np.float64)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def adaptive_alpha(v: np.ndarray, a: float, b: float) -> float:
    """alpha = min(1, max(0, a + b * mu_V)), mu_V pada skala 0..1 (Persamaan 3)."""
    mu = float(v.mean()) / 255.0
    return float(np.clip(a + b * mu, 0.0, 1.0))


def gamma_channel(v: np.ndarray, gamma: float) -> np.ndarray:
    """Koreksi gamma sederhana dengan tabel lookup."""
    table = np.clip(np.round(255.0 * (np.arange(256) / 255.0) ** gamma), 0, 255)
    return cv2.LUT(v, table.astype(np.uint8))


# ------------------------------------------------------------ pipeline
def enhance(img_rgb, method: str, clip_limit: float = 2.0, tile: int = 8,
            sigma: float = 2.0, alpha: float = 0.5, a: float = 0.5,
            b: float = 0.0, gamma: float = 0.5):
    """
    Terapkan satu metode pada citra RGB.
    Mengembalikan (citra_hasil_rgb_uint8, alpha_yang_dipakai_atau_None).
    """
    img = ensure_rgb_uint8(img_rgb)
    if method == "asli":
        return img.copy(), None

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    used_alpha = None

    if method == "he":
        nv = he_channel(v)
    elif method == "clahe":
        nv = clahe_channel(v, clip_limit, tile)
    elif method == "bpdhe":
        nv = bpdhe_channel(v, sigma)
    elif method == "kombinasi_tetap":
        used_alpha = float(alpha)
        nv = blend_channel(clahe_channel(v, clip_limit, tile),
                           bpdhe_channel(v, sigma), used_alpha)
    elif method == "kombinasi_adaptif":
        used_alpha = adaptive_alpha(v, a, b)
        nv = blend_channel(clahe_channel(v, clip_limit, tile),
                           bpdhe_channel(v, sigma), used_alpha)
    elif method == "gamma":
        nv = gamma_channel(v, gamma)
    else:
        raise ValueError(f"Metode tidak dikenal: {method}")

    out = cv2.cvtColor(cv2.merge([h, s, nv]), cv2.COLOR_HSV2RGB)
    return out, used_alpha


# ------------------------------------------------------------- metrik
def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def entropy(gray: np.ndarray) -> float:
    """Entropi citra dalam bit (Persamaan 7)."""
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def ambe(x: np.ndarray, y: np.ndarray) -> float:
    """Selisih mutlak rata-rata kecerahan (Persamaan 8)."""
    return float(abs(_gray(x).mean() - _gray(y).mean()))


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    """PSNR dalam dB (Persamaan 4 dan 5), dibatasi PSNR_CAP bila identik."""
    mse = np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)
    if mse == 0:
        return PSNR_CAP
    return float(min(10 * np.log10(255.0 ** 2 / mse), PSNR_CAP))


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    """SSIM pada citra keabuan (Persamaan 6)."""
    return float(structural_similarity(_gray(x), _gray(y), data_range=255))


def evaluate(orig: np.ndarray, out: np.ndarray) -> dict:
    """
    Metrik per citra. PSNR dan SSIM di sini diukur terhadap MASUKAN
    (ExDark tidak punya citra acuan), sehingga tidak dibaca sebagai mutu tunggal.
    """
    orig = ensure_rgb_uint8(orig)
    out = ensure_rgb_uint8(out)
    return {
        "Entropi": entropy(_gray(out)),
        "SSIM (thd. masukan)": ssim(orig, out),
        "PSNR dB (thd. masukan)": psnr(orig, out),
        "AMBE": ambe(orig, out),
        "Kenaikan kecerahan": float(_gray(out).mean() - _gray(orig).mean()),
    }


# ------------------------------------------------------ contoh gelap
def sample_dark_image(factor: float = 0.15, noise_sigma: float = 4.0,
                      seed: int = 7) -> np.ndarray:
    """
    Contoh citra gelap sintetis dari gambar bawaan scikit-image (tanpa unduhan),
    dibuat dengan menurunkan kecerahan dan menambahkan noise. Hanya untuk demo.
    """
    from skimage import data

    base = data.chelsea().astype(np.float64)
    rng = np.random.default_rng(seed)
    dark = base * factor + rng.normal(0.0, noise_sigma, base.shape)
    return np.clip(dark, 0, 255).astype(np.uint8)
