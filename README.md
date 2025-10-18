# 🪄 Auto Pas Foto Generator (Background Merah + Resize 3x4)

Sistem ini membantu kamu mengedit hasil foto siswa dari photoshoot menjadi **ukuran pas foto 3x4** dengan **background merah otomatis**, serta **menyamakan tone warna** agar hasilnya lebih rapi dan seragam.  
Selain itu sistem ini juga mendukung **auto rename file menggunakan NIS (Nomor Induk Siswa)** dan **batch log** agar proses dokumentasi lebih efisien.

---

## ✨ Fitur Utama
- 🖼️ **Auto Remove Background** (pakai AI)  
- 🟥 Ganti background otomatis ke warna merah (kode warna bisa disesuaikan)  
- 🧑‍🎓 Resize ke ukuran pas foto standar 3x4 cm  
- 🧼 Koreksi warna otomatis (hilangkan kebiruan / kekuningan berlebihan)  
- 🪪 Auto rename file berdasarkan NIS siswa  
- 🧾 Generate batch log (riwayat proses file)

---

## 🛠️ Tech Stack
- Python 3.10+
- :contentReference[oaicite:1]{index=1} (background remover)
- :contentReference[oaicite:2]{index=2} (image processing)
- :contentReference[oaicite:3]{index=3} (array & pixel ops)
- :contentReference[oaicite:4]{index=4} (auto color correction — opsional)

---

## 📦 Cara Instalasi & Setup

### 1. Clone Repository
```bash
git clone https://github.com/username/auto-pasfoto-generator.git
cd auto-pasfoto-generator

