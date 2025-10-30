# auto_pasfoto_fix.py
import os
import io
import math
import cv2
import numpy as np
from PIL import Image
from rembg import remove
import mediapipe as mp

# ---------- Konfigurasi ----------
BACKGROUND_COLOR = (205, 18, 15)   # #cd120f
WIDTH_PX = int(3 * 300 / 2.54)     # 354
HEIGHT_PX = int(4 * 300 / 2.54)    # 472
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Tweakable params
FACE_SCALE_W = 2.4
UPSHIFT = 0.45
ALPHA_BLUR_RATIO = 0.02   # semakin besar -> tepi lebih lembut (0.01 - 0.04)
MIN_FACE_W = 30

mp_face_mesh = mp.solutions.face_mesh

# ---------- Util: rotate with expanded canvas ----------
def rotate_expand(img, angle_deg, center=None):
    h, w = img.shape[:2]
    if center is None:
        center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0,2] += (new_w/2) - center[0]
    M[1,2] += (new_h/2) - center[1]
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return rotated, M

# ---------- Util: refine alpha mask ----------
def refine_alpha(pil_rgba):
    arr = np.array(pil_rgba)  # HxWx4
    if arr.shape[2] < 4:
        # no alpha -> return pil_rgba directly
        return pil_rgba
    alpha = arr[:,:,3]
    h, w = alpha.shape
    # kernel size proportional to image diag
    k = max(3, int(max(w,h) * ALPHA_BLUR_RATIO))
    if k % 2 == 0:
        k += 1
    # smooth alpha
    alpha_blur = cv2.GaussianBlur(alpha, (k,k), 0)
    # normalize & recompose
    fg_rgb = arr[:,:,0:3]
    alpha_norm = (alpha_blur / 255.0)[:,:,None]
    comp = (fg_rgb * alpha_norm + 255 * (1 - alpha_norm)).astype(np.uint8)
    # create RGBA pil
    out_arr = np.dstack([comp[:,:,0], comp[:,:,1], comp[:,:,2], alpha_blur])
    return Image.fromarray(out_arr, mode="RGBA")

# ---------- Core ----------
def process_image(path_in, path_out):
    img_bgr = cv2.imread(path_in)
    if img_bgr is None:
        print("❌ Tidak bisa baca:", path_in); return False
    h0, w0 = img_bgr.shape[:2]

    # first pass: detect face landmarks to compute rotation angle using FaceMesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(img_rgb)
        if not res.multi_face_landmarks:
            print("⚠️ Wajah tidak terdeteksi (awal):", os.path.basename(path_in))
            # fallback: no rotate, use center crop
            angle = 0.0
            rotated = img_bgr.copy()
            rot_center = (w0//2, h0//2)
        else:
            lm = res.multi_face_landmarks[0].landmark
            # average eye points for stability (left eye indices cluster & right)
            left_idxs = [33, 133, 160, 159]
            right_idxs = [263, 362, 387, 386]
            def avg_point(idxs):
                xs=[]; ys=[]
                for i in idxs:
                    xs.append(lm[i].x * w0)
                    ys.append(lm[i].y * h0)
                return (sum(xs)/len(xs), sum(ys)/len(ys))
            left_eye = avg_point(left_idxs)
            right_eye = avg_point(right_idxs)
            dx = right_eye[0] - left_eye[0]; dy = right_eye[1] - left_eye[1]
            angle = math.degrees(math.atan2(dy, dx))
            mid = (int((left_eye[0]+right_eye[0])/2), int((left_eye[1]+right_eye[1])/2))
            rotated, M = rotate_expand(img_bgr, -angle, center=mid)
            rot_center = mid

    # second pass: detect face landmarks on rotated image to get accurate bbox
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm2:
        rgb2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        res2 = fm2.process(rgb2)
        h1, w1 = rotated.shape[:2]
        if not res2.multi_face_landmarks:
            # fallback center crop
            cx, cy = w1//2, h1//2
            face_w = int(min(w1,h1) * 0.2)
            print("⚠️ Wajah tidak terdeteksi setelah rotasi (fallback crop):", os.path.basename(path_in))
        else:
            lm2 = res2.multi_face_landmarks[0].landmark
            pts = [(int(p.x * w1), int(p.y * h1)) for p in lm2]
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            xmin, xmax = max(0,min(xs)), min(w1-1,max(xs))
            ymin, ymax = max(0,min(ys)), min(h1-1,max(ys))
            face_w = xmax - xmin
            cx = (xmin + xmax)//2
            cy = (ymin + ymax)//2

    if face_w < MIN_FACE_W:
        face_w = max(MIN_FACE_W, int(min(w1,h1) * 0.2))

    # compute crop size 3:4
    ratio = HEIGHT_PX / WIDTH_PX
    crop_w = int(face_w * FACE_SCALE_W)
    crop_h = int(crop_w * ratio)

    x1 = max(0, cx - crop_w//2)
    y1 = max(0, int(cy - crop_h * UPSHIFT))
    x2 = min(w1, x1 + crop_w)
    y2 = min(h1, y1 + crop_h)
    # adjust if clipped
    if (x2 - x1) < crop_w:
        x1 = max(0, x2 - crop_w)
    if (y2 - y1) < crop_h:
        y1 = max(0, y2 - crop_h)

    crop = rotated[y1:y2, x1:x2].copy()
    if crop.size == 0:
        print("❌ Crop kosong:", path_in)
        return False

    # remove background using rembg
    _, enc = cv2.imencode('.png', crop)
    try:
        bytes_no_bg = remove(enc.tobytes())
    except Exception as e:
        print("⚠️ rembg failed:", e, "→ save raw crop")
        pil_raw = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGBA")
        pil_raw = pil_raw.resize((WIDTH_PX, HEIGHT_PX), Image.LANCZOS)
        bg = Image.new("RGBA", (WIDTH_PX, HEIGHT_PX), BACKGROUND_COLOR + (255,))
        out = Image.alpha_composite(bg, pil_raw)
        out.convert("RGB").save(path_out, "JPEG", quality=100)
        return True

    pil_fg = Image.open(io.BytesIO(bytes_no_bg)).convert("RGBA")
    # refine alpha edges
    pil_refined = refine_alpha(pil_fg)

    # resize to final size
    # pil_resized = pil_refined.resize((WIDTH_PX, HEIGHT_PX), Image.LANCZOS)

    # compose on solid background with soft edges
    # final_bg = Image.new("RGBA", (WIDTH_PX, HEIGHT_PX), BACKGROUND_COLOR + (255,))
    # final = Image.alpha_composite(final_bg, pil_resized)
    # final.convert("RGB").save(path_out, "JPEG", quality=100)
    
    # Ambil bounding box area non-transparan
    bbox = pil_refined.getbbox()
    if bbox:
        pil_refined = pil_refined.crop(bbox)

    # Ukuran background akhir (3x4 cm @300dpi)
    bg_width, bg_height = WIDTH_PX, HEIGHT_PX
    bg = Image.new("RGBA", (bg_width, bg_height), BACKGROUND_COLOR + (255,))

    # Resize foreground biar proporsional di tengah (90% ukuran background)
    scale = min(bg_width / pil_refined.width * 0.9, bg_height / pil_refined.height * 0.9)
    new_size = (int(pil_refined.width * scale), int(pil_refined.height * scale))
    pil_resized = pil_refined.resize(new_size, Image.LANCZOS)

    # Hitung posisi tengah yang presisi
    offset = ((bg_width - pil_resized.width) // 2, (bg_height - pil_resized.height) // 2)

    # Tempelkan ke background merah
    bg.paste(pil_resized, offset, pil_resized)
    bg.convert("RGB").save(path_out, "JPEG", quality=100)
    print(f"✅ {os.path.basename(path_in)} -> {os.path.basename(path_out)} (rot {angle:.1f}°)")
    return True

# ---------- Main ----------
if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg",".jpeg",".png"))]
    if not files:
        print("Folder input kosong. Masukkan file gambar ke folder 'input/'.")
    for f in files:
        in_path = os.path.join(INPUT_FOLDER, f)
        out_path = os.path.join(OUTPUT_FOLDER, f"pasfoto_{f}")
        try:
            process_image(in_path, out_path)
        except Exception as e:
            print("❌ Error processing", f, ":", e)
