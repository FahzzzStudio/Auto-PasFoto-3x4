Auto-Pasfoto

Sistem semi-otomatis untuk mengubah foto hasil photoshoot menjadi pas foto ukuran 3×4 cm dengan background merah (kode warna #cd120f) dan wajah di-tengah.
Fokus: crop, hapus background, ganti background merah, preserve kualitas. Tidak melakukan normalisasi warna otomatis.

Fitur

Deteksi wajah otomatis (Haar Cascade)

Crop dengan framing 3×4 berdasarkan posisi wajah

Hapus background otomatis (rembg + model u2net)

Ganti background dengan warna merah #cd120f

Simpan hasil dengan kualitas maksimum (JPEG quality 100)

Batch processing (map input/ → output/)

Struktur proyek (disarankan)
auto-pasfoto/
├─ input/                 # masukkan foto mentah di sini
├─ output/                # hasil akan disimpan di sini
├─ auto_pasfoto.py        # script utama
├─ requirements.txt
├─ data_siswa.csv         # (opsional) untuk auto-rename NIS
├─ batch_log.txt          # (akan dibuat otomatis)
├─ README.md
└─ .gitignore

Contoh requirements.txt

Letakkan file ini agar orang gampang install:

opencv-python
numpy
pillow
rembg
scikit-image
onnxruntime      # atau onnxruntime-gpu jika pakai GPU


Catatan: rembg membutuhkan onnxruntime karena model u2net menggunakan ONNX.

Cara instal & jalankan (Windows / macOS / Linux)

Berikan instruksi step-by-step untuk pemula.

1. Clone repo
git clone https://github.com/<username>/auto-pasfoto.git
cd auto-pasfoto

2. Buat virtual environment (direkomendasikan)

Windows (PowerShell)

python -m venv venv
.\venv\Scripts\Activate.ps1
# jika muncul policy error, jalankan PowerShell sebagai administrator lalu:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


Windows (cmd)

python -m venv venv
venv\Scripts\activate


macOS / Linux

python3 -m venv venv
source venv/bin/activate


Kamu akan melihat prompt berubah, misal (venv) C:\...>.

3. Update pip (opsional tapi direkomendasikan)
pip install --upgrade pip

4. Install dependency

Pastikan requirements.txt ada, lalu:

pip install -r requirements.txt


Jika kamu tidak punya requirements.txt, install manual:

pip install opencv-python numpy pillow rembg scikit-image onnxruntime


Bila pakai GPU dan ingin onnxruntime GPU build:

pip install onnxruntime-gpu

5. Struktur folder input/output

Buat folder input dan output:

mkdir input output


Taruh foto mentah (JPEG/PNG) ke folder input.

6. (Opsional) data_siswa.csv untuk auto rename

Contoh data_siswa.csv (jika menambahkan fitur rename):

nama_file_asli,nis,nama
DSC09841.JPG,2301980,Ani Kusuma
DSC09842.JPG,2301981,Budi Setiawan

7. Jalankan script
python auto_pasfoto.py


Hasil akan muncul di output/ dengan nama pasfoto_<originalname>.jpg (atau sesuai logic auto-rename kalau diimplementasikan).

Script (contoh minimal tanpa normalisasi warna — gunakan file auto_pasfoto.py)

Gunakan script ini (sudah disesuaikan dengan permintaanmu: background #cd120f, tidak ada color correction, quality 100):

import os
import cv2
from rembg import remove
from PIL import Image
import io

# Konfigurasi
BACKGROUND_COLOR = (205, 18, 15)  # #cd120f
WIDTH_PX = int(3 * 300 / 2.54)   # 354 px
HEIGHT_PX = int(4 * 300 / 2.54)  # 472 px
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def process_photo(image_path, output_path):
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"❌ Tidak bisa membaca: {image_path}")
        return

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"❌ Tidak ada wajah terdeteksi: {os.path.basename(image_path)}")
        return

    x, y, w, h = faces[0]
    cx, cy = x + w // 2, y + h // 2

    ratio = HEIGHT_PX / WIDTH_PX
    crop_w = int(w * 2.2)
    crop_h = int(crop_w * ratio)
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - int(crop_h * 0.45))
    x2 = min(img_cv.shape[1], x1 + crop_w)
    y2 = min(img_cv.shape[0], y1 + crop_h)
    crop_img = img_cv[y1:y2, x1:x2]

    # Hapus background
    _, encoded_img = cv2.imencode('.png', crop_img)
    img_no_bg = remove(encoded_img.tobytes())

    # Convert ke PIL
    img = Image.open(io.BytesIO(img_no_bg)).convert("RGBA")

    # Resize ke 3x4
    img = img.resize((WIDTH_PX, HEIGHT_PX), Image.LANCZOS)

    # Background merah
    bg = Image.new("RGBA", (WIDTH_PX, HEIGHT_PX), BACKGROUND_COLOR + (255,))

    final = Image.alpha_composite(bg, img)
    final.convert("RGB").save(output_path, "JPEG", quality=100)
    print(f"✅ {os.path.basename(image_path)} → {os.path.basename(output_path)}")

if __name__ == "__main__":
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, f"pasfoto_{filename}")
            process_photo(input_path, output_path)

Troubleshooting umum

ModuleNotFoundError: No module named 'onnxruntime'
→ pip install onnxruntime (atau onnxruntime-gpu untuk GPU).

Masalah rembg mendownload model u2net
→ rembg pertama kali jalan akan mendownload ~176MB model u2net. Pastikan koneksi. File akan disimpan di ~/.u2net/u2net.onnx.

Error AttributeError: 'bytearray' object has no attribute 'read'
→ Pastikan kamu convert bytes ke io.BytesIO() sebelum Image.open() seperti di contoh.

Tidak ada wajah terdeteksi
→ Pastikan foto jelas, wajah tidak terlalu kecil di frame. Kamu bisa tweak scaleFactor / minNeighbors atau pakai DNN face detector untuk akurasi lebih baik.

Warna menjadi aneh / gelap / hitam
→ Pastikan kamu tidak menggunakan step normalisasi warna. Script di README sudah tanpa normalisasi. Jika tetap aneh, cek apakah rembg menghasilkan alpha mask yang tidak rapi di tepi — coba gunakan foto dengan kontras yang baik atau edit manual sedikit di Photoshop seperti yang kamu inginkan.

Menjaga kualitas gambar

PIL.save(..., quality=100) untuk JPEG maksimal.

Jika ingin output PNG (tanpa kompresi lossy): ganti save(..., "PNG").
