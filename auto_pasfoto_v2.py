import os
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from PIL import Image
import io

# ==================== KONFIGURASI ====================
BACKGROUND_COLOR = (205, 18, 15)  # Merah dalam RGB
WIDTH_PX = int(3 * 300 / 2.54)   # 3 cm @300dpi
HEIGHT_PX = int(4 * 300 / 2.54)  # 4 cm @300dpi
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# ==================== FUNGSI BANTUAN ====================
def rotate_image(image, angle, center=None, scale=1.0):
    """Rotasi gambar agar kepala tegak"""
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

# ==================== FUNGSI UTAMA ====================
def process_photo(image_path, output_path):
    img_cv = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_rgb)

        if not results.detections:
            print(f"❌ Tidak ada wajah terdeteksi: {image_path}")
            return

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = img_cv.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        # ==== Koreksi rotasi kepala ====
        keypoints = detection.location_data.relative_keypoints
        left_eye = keypoints[0]
        right_eye = keypoints[1]

        eye_dx = (right_eye.x - left_eye.x) * w
        eye_dy = (right_eye.y - left_eye.y) * h
        angle = np.degrees(np.arctan2(eye_dy, eye_dx))

        img_rotated = rotate_image(img_cv, angle)
        gray_rotated = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)

        # ==== Crop ulang setelah rotasi ====
        x1 = max(0, x - int(bw * 0.3))
        y1 = max(0, y - int(bh * 0.4))
        x2 = min(w, x + int(bw * 1.3))
        y2 = min(h, y + int(bh * 1.5))
        crop_img = img_rotated[y1:y2, x1:x2]

        # ==== Hapus background ====
        _, encoded_img = cv2.imencode('.png', crop_img)
        img_no_bg = remove(encoded_img.tobytes())

        img = Image.open(io.BytesIO(img_no_bg)).convert("RGBA")
        img = img.resize((WIDTH_PX, HEIGHT_PX), Image.LANCZOS)

        bg = Image.new("RGBA", (WIDTH_PX, HEIGHT_PX), BACKGROUND_COLOR + (255,))
        final = Image.alpha_composite(bg, img)

        final.convert("RGB").save(output_path, "JPEG", quality=100)
        print(f"✅ Berhasil: {os.path.basename(image_path)}")

# ==================== MAIN LOOP ====================
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"pasfoto_{filename}")
        process_photo(input_path, output_path)
