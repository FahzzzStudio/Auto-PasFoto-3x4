import os
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import io

# =============== Konfigurasi ===============
BACKGROUND_COLOR = (205, 18, 15)  # Merah #cd120f dalam RGB
WIDTH_PX = int(3 * 300 / 2.54)   # 354 px
HEIGHT_PX = int(4 * 300 / 2.54)  # 472 px
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Model Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# =============== Fungsi Utama ===============
def process_photo(image_path, output_path):
    # 1. Baca gambar
    img_cv = cv2.imread(image_path)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 2. Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print(f"❌ Tidak ada wajah terdeteksi: {image_path}")
        return

    (x, y, w, h) = faces[0]
    cx, cy = x + w // 2, y + h // 2

    # 3. Crop foto dengan rasio 3x4 di sekitar wajah
    ratio = HEIGHT_PX / WIDTH_PX
    crop_w = int(w * 2.2)
    crop_h = int(crop_w * ratio)
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - int(crop_h * 0.45))
    x2 = min(img_cv.shape[1], x1 + crop_w)
    y2 = min(img_cv.shape[0], y1 + crop_h)
    crop_img = img_cv[y1:y2, x1:x2]

    # 4. Hapus background pakai rembg
    _, encoded_img = cv2.imencode('.png', crop_img)
    img_no_bg = remove(encoded_img.tobytes())

    # 5. Convert ke Pillow pakai io.BytesIO
    img = Image.open(io.BytesIO(img_no_bg)).convert("RGBA")

    # 6. Resize ke ukuran 3x4 cm
    img = img.resize((WIDTH_PX, HEIGHT_PX), Image.LANCZOS)

    # 7. Buat background merah polos
    bg = Image.new("RGBA", (WIDTH_PX, HEIGHT_PX), BACKGROUND_COLOR + (255,))

    # 8. Tempel objek ke background
    final = Image.alpha_composite(bg, img)

    # 9. Simpan hasil
    final.convert("RGB").save(output_path, "JPEG", quality=100)  # 100 biar tajem maksimal
    print(f"✅ Berhasil: {os.path.basename(image_path)}")

# =============== Main Loop ===============
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"pasfoto_{filename}")
        process_photo(input_path, output_path)
