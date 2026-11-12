"""
Kalibrasi bobot kombinasi CLAHE + BPDHE pada SET TUNING (Bagian 7.3 naskah).

Cara pakai (di komputer/Colab yang punya folder citra tuning):
    python calibrate.py --folder path/ke/set_tuning --out calibration.json

Yang dilakukan:
1. Untuk setiap citra, coba alpha 0.0 sampai 1.0 (langkah 0.1).
2. Pilih alpha terbaik per citra dengan skor gabungan = rata-rata peringkat
   Entropi (makin tinggi makin baik), SSIM (makin tinggi makin baik),
   dan AMBE (makin rendah makin baik).
3. Bobot tetap = alpha yang rata-rata skor gabungannya tertinggi di seluruh set.
4. Bobot adaptif: regresi linear alpha_terbaik ~ mu_V, sehingga a dan b
   dapat dipakai di Persamaan (3).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy.stats import rankdata

from lowlight import (blend_channel, bpdhe_channel, clahe_channel,
                      ensure_rgb_uint8, evaluate)

ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 1)
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")


def composite_scores(img_rgb: np.ndarray, clip: float, tile: int, sigma: float):
    """Skor gabungan untuk setiap alpha pada satu citra, dan mu_V-nya."""
    img = ensure_rgb_uint8(img_rgb)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    v1 = clahe_channel(v, clip, tile)
    v2 = bpdhe_channel(v, sigma)

    ent, ssi, amb = [], [], []
    for a in ALPHAS:
        nv = blend_channel(v1, v2, a)
        out = cv2.cvtColor(cv2.merge([h, s, nv]), cv2.COLOR_HSV2RGB)
        m = evaluate(img, out)
        ent.append(m["Entropi"])
        ssi.append(m["SSIM (thd. masukan)"])
        amb.append(m["AMBE"])

    # Peringkat: nilai tinggi diberi peringkat tinggi, AMBE dibalik.
    r_ent = rankdata(ent)
    r_ssi = rankdata(ssi)
    r_amb = rankdata([-x for x in amb])
    composite = (r_ent + r_ssi + r_amb) / 3.0
    mu = float(v.mean()) / 255.0
    return composite, mu


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", required=True, help="folder set tuning")
    parser.add_argument("--out", default="calibration.json")
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--tile", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=2.0)
    args = parser.parse_args()

    files = sorted(f for ext in EXTS for f in glob.glob(os.path.join(args.folder, ext)))
    if not files:
        raise SystemExit("Tidak ada citra di folder tersebut.")

    comp_all, best_alphas, mus = [], [], []
    for path in files:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:  # berkas rusak dikeluarkan (Bagian 6.3)
            print(f"[lewati] tidak bisa dibaca: {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        comp, mu = composite_scores(rgb, args.clip, args.tile, args.sigma)
        comp_all.append(comp)
        best_alphas.append(float(ALPHAS[int(np.argmax(comp))]))
        mus.append(mu)
        print(f"{os.path.basename(path)}: mu={mu:.3f} alpha_terbaik={best_alphas[-1]:.1f}")

    if not comp_all:
        raise SystemExit("Tidak ada citra yang valid.")

    comp_mean = np.mean(np.vstack(comp_all), axis=0)
    alpha_fixed = float(ALPHAS[int(np.argmax(comp_mean))])

    mus_arr, al_arr = np.array(mus), np.array(best_alphas)
    if len(mus_arr) >= 2 and np.std(mus_arr) > 1e-9:
        b, a = np.polyfit(mus_arr, al_arr, 1)
        r = float(np.corrcoef(mus_arr, al_arr)[0, 1])
    else:
        a, b, r = float(al_arr.mean()), 0.0, 0.0

    result = {
        "alpha_fixed": alpha_fixed,
        "a": round(float(a), 4),
        "b": round(float(b), 4),
        "korelasi_r": round(r, 4),
        "jumlah_citra_tuning": len(comp_all),
        "clip_limit": args.clip,
        "tile": args.tile,
        "sigma": args.sigma,
        "catatan": "Hasil kalibrasi pada set tuning, bukan set uji.",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
