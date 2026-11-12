# Peningkatan Kualitas Citra Cahaya Rendah (Demo Final Project)

Kombinasi CLAHE dan BPDHE pada dataset ExDark, dibangun dengan Streamlit.
Aplikasi ini menyajikan metode HE, CLAHE, BPDHE, kombinasi bobot tetap dan adaptif,
gamma, serta metrik evaluasi (entropi, SSIM, PSNR, AMBE, kenaikan kecerahan).

## Isi folder

| Berkas | Fungsi |
|---|---|
| `app.py` | Antarmuka Streamlit (demo) |
| `lowlight.py` | Algoritma: HE, CLAHE, BPDHE, kombinasi, gamma, metrik |
| `calibrate.py` | Kalibrasi bobot α tetap dan koefisien adaptif a, b pada set tuning |
| `calibration.json` | Nilai bobot yang dibaca aplikasi (saat ini nilai default) |
| `requirements.txt` | Daftar pustaka Python |

## 1. Jalankan di localhost (laptop/PC)

```bash
# buat lingkungan virtual (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Buka alamat yang muncul di terminal, biasanya **http://localhost:8501**.

## 2. Deploy gratis di Streamlit Community Cloud (bisa dibuka dari HP)

1. Unggah folder ini ke repositori GitHub publik (bisa lewat antarmuka web GitHub di browser HP).
2. Buka https://share.streamlit.io, login dengan akun GitHub.
3. Klik **Create app**, pilih repositori, branch `main`, dan file utama `app.py`.
4. Klik **Deploy**. Setelah selesai, Anda mendapat link publik yang bisa dibuka dari HP dosen.

## 3. Kalibrasi bobot adaptif (opsional tapi disarankan untuk hasil final)

Nilai `a`, `b`, dan `alpha_fixed` di `calibration.json` harus dihitung dari **set tuning**
(20% citra ExDark, sesuai Bagian 6.3 naskah):

```bash
python calibrate.py --folder path/ke/set_tuning --out calibration.json
```

Setelah itu, aplikasi otomatis memakai nilai baru saat dijalankan ulang.

## Catatan penting

- Demo ini memakai **PSNR dan SSIM terhadap citra masukan**, karena ExDark tidak
  memiliki citra acuan. Lihat penjelasan di dalam aplikasi.
- **NIQE** dan validasi pada dataset **LOL** tidak disertakan di demo. Keduanya berada
  di skrip eksperimen penuh.
- Aplikasi memperkecil citra ke sisi terpanjang 900 piksel agar cepat dijalankan.
- Teknik restorasi warna dari Thepade dkk. (2022) belum diimplementasikan, karena naskah
  lengkapnya belum dibaca. Pemrosesan dilakukan pada kanal V dari HSV.
- Nilai default `calibration.json` belum dikalibrasi pada data ExDark. Jangan menyebutnya
  sebagai hasil final sebelum calibrate.py dijalankan.
