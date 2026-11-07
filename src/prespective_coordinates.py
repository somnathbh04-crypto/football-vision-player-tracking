import cv2
import numpy as np


# ==========================================
# 1. OPEN VIDEO AND GET ONE FRAME
# ==========================================

video_path = "VIDEOS/13386254_3840_2160_24fps.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    exit()

success, frame = cap.read()
cap.release()

if not success:
    print("Could not read frame.")
    exit()


# Original frame size
original_height, original_width = frame.shape[:2]

print("Original size:", original_width, "x", original_height)


# ==========================================
# 2. RESIZE ONLY FOR DISPLAY
# ==========================================

display_width = 1000
display_height = 563

display_frame = cv2.resize(
    frame,
    (display_width, display_height)
)


# Scale factors
scale_x = original_width / display_width
scale_y = original_height / display_height


# ==========================================
# 3. SELECT FOUR POINTS
# ==========================================

points = []


def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        # Coordinates on the resized display
        print("Clicked display point:", (x, y))

        # Convert to ORIGINAL image coordinates
        original_x = int(x * scale_x)
        original_y = int(y * scale_y)

        points.append((original_x, original_y))

        print(
            "Original image point:",
            (original_x, original_y)
        )

        # Draw the point on the displayed image
        cv2.circle(
            display_frame,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display_frame,
            str(len(points)),
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


cv2.namedWindow(
    "SELECT PITCH POINTS",
    cv2.WINDOW_AUTOSIZE
)

cv2.setMouseCallback(
    "SELECT PITCH POINTS",
    mouse_callback
)


print("\nClick four points in this order:")
print("1 = top-left")
print("2 = top-right")
print("3 = bottom-right")
print("4 = bottom-left")
print("\nPress Q when finished.")


while True:

    cv2.imshow(
        "SELECT PITCH POINTS",
        display_frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cv2.destroyAllWindows()


# ==========================================
# 4. CHECK POINTS
# ==========================================

if len(points) != 4:

    print(
        "You selected",
        len(points),
        "points. Exactly 4 are required."
    )

    exit()


print("\nOriginal image coordinates:")
for i, point in enumerate(points, start=1):
    print(f"Point {i}: {point}")


# ==========================================
# 5. PERSPECTIVE TRANSFORMATION
# ==========================================

src_points = np.float32(points)

dst_points = np.float32([
    [0, 0],
    [1000, 0],
    [1000, 600],
    [0, 600]
])


matrix = cv2.getPerspectiveTransform(
    src_points,
    dst_points
)


top_view = cv2.warpPerspective(
    frame,
    matrix,
    (1000, 600)
)


# ==========================================
# 6. SHOW RESULT
# ==========================================

cv2.namedWindow(
    "TOP VIEW",
    cv2.WINDOW_NORMAL
)

cv2.imshow(
    "TOP VIEW",
    top_view
)

print("\nPress any key to close the top view.")

cv2.waitKey(0)
cv2.destroyAllWindows()
