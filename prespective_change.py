import cv2


# ==========================================
# 1. OPEN VIDEO
# ==========================================

video_path = "VIDEOS/13386254_3840_2160_24fps.mp4"

cap = cv2.VideoCapture(video_path)


# ==========================================
# 2. CHECK VIDEO
# ==========================================

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

print("Video opened successfully.")


# ==========================================
# 3. READ ONE FRAME
# ==========================================

success, frame = cap.read()

cap.release()


if not success or frame is None:
    print("ERROR: Could not read the first frame.")
    exit()

print("Frame read successfully.")
print("Frame size:", frame.shape)


# ==========================================
# 4. SHOW FRAME
# ==========================================

cv2.namedWindow(
    "Football Frame",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Football Frame",
    1200,
    700
)

cv2.imshow(
    "Football Frame",
    frame
)

print("Press any key inside the image window to close it.")

cv2.waitKey(0)

cv2.destroyAllWindows()