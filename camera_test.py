import math
import os

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.tasks.vision.drawing_utils
POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

EKSTRA_NOKTA_BAGLANTILARI = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (11, 12),
    (23, 24),
    (11, 23), (12, 24),
]
ARA_NOKTA_SAYISI = 3
ARA_NOKTA_RENGI = (0, 255, 255)
ARA_NOKTA_YARICAP = 3

def ara_noktalari_ciz(image, pose_landmarks, connections, ara_nokta_sayisi):
    image_h, image_w = image.shape[:2]
    for start_idx, end_idx in connections:
        start_lm = pose_landmarks[start_idx]
        end_lm = pose_landmarks[end_idx]
        if start_lm.visibility is not None and start_lm.visibility < 0.5:
            continue
        if end_lm.visibility is not None and end_lm.visibility < 0.5:
            continue
        for i in range(1, ara_nokta_sayisi + 1):
            t = i / (ara_nokta_sayisi + 1)
            x = start_lm.x + (end_lm.x - start_lm.x) * t
            y = start_lm.y + (end_lm.y - start_lm.y) * t
            px, py = int(x * image_w), int(y * image_h)
            cv2.circle(image, (px, py), ARA_NOKTA_YARICAP, ARA_NOKTA_RENGI, -1)


SOL_KULAK, SAG_KULAK, BURUN = 7, 8, 0
SOL_OMUZ, SAG_OMUZ = 11, 12
KAFA_BOY_EN_ORANI = 1.3
KAFA_GENISLIK_PAYI = 1.25
KAFA_RENGI = (255, 0, 255)
BOYUN_RENGI = (255, 0, 255)


def kafa_ve_boyun_ciz(image, pose_landmarks):
    image_h, image_w = image.shape[:2]
    gerekli_idx = (SOL_KULAK, SAG_KULAK, BURUN, SOL_OMUZ, SAG_OMUZ)
    gerekli_lm = [pose_landmarks[i] for i in gerekli_idx]
    if any(lm.visibility is not None and lm.visibility < 0.5 for lm in gerekli_lm):
        return

    sol_kulak_lm, sag_kulak_lm, burun_lm, sol_omuz_lm, sag_omuz_lm = gerekli_lm

    def piksel(lm):
        return int(lm.x * image_w), int(lm.y * image_h)

    sol_kulak_px = piksel(sol_kulak_lm)
    sag_kulak_px = piksel(sag_kulak_lm)
    burun_px = piksel(burun_lm)
    omuz_orta_px = (
        int((sol_omuz_lm.x + sag_omuz_lm.x) / 2 * image_w),
        int((sol_omuz_lm.y + sag_omuz_lm.y) / 2 * image_h),
    )

    kulak_mesafesi = math.hypot(sag_kulak_px[0] - sol_kulak_px[0],
                                 sag_kulak_px[1] - sol_kulak_px[1])
    kafa_genisligi = max(int(kulak_mesafesi * KAFA_GENISLIK_PAYI), 1)
    kafa_uzunlugu = max(int(kafa_genisligi * KAFA_BOY_EN_ORANI), 1)

    kafa_merkezi = (
        (sol_kulak_px[0] + sag_kulak_px[0]) // 2,
        burun_px[1] - kafa_uzunlugu // 6,
    )

    cv2.ellipse(
        image,
        kafa_merkezi,
        (kafa_genisligi // 2, kafa_uzunlugu // 2),
        0, 0, 360,
        KAFA_RENGI, 2,
    )

    kafa_alti_px = (kafa_merkezi[0], kafa_merkezi[1] + kafa_uzunlugu // 2)
    cv2.line(image, kafa_alti_px, omuz_orta_px, BOYUN_RENGI, 2)


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_landmarker_lite.task')

if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        f"HATA: Model dosyasi bulunamadi: {MODEL_PATH}\n"
        "'pose_landmarker_lite.task' dosyasinin camera_test.py ile ayni "
        "klasorde oldugundan emin olun."
    )

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
landmarker = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("HATA: Kamera acilamadi. Baska bir uygulamanin kamerayi kullanip "
          "kullanmadigini veya kamera index'inin dogru oldugunu kontrol edin.")
    landmarker.close()
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

PENCERE_ADI = "Poz Algilama - Kamera Testi"
cv2.namedWindow(PENCERE_ADI, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(PENCERE_ADI, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Kamera acildi. Cikmak icin 'q' tusuna basin.")

frame_index = 0

while cap.isOpened():
    basarili, frame = cap.read()
    if not basarili:
        print("UYARI: Kameradan kare okunamadi, dongu sonlandiriliyor.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(frame_index * (1000 / 30))
    frame_index += 1

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        for pose_landmarks in result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                POSE_CONNECTIONS,
            )
            ara_noktalari_ciz(frame, pose_landmarks, EKSTRA_NOKTA_BAGLANTILARI, ARA_NOKTA_SAYISI)
            kafa_ve_boyun_ciz(frame, pose_landmarks)

    cv2.imshow(PENCERE_ADI, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Cikis yapiliyor...")
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
