import cv2
import numpy as np

video_path = "VIDEOS/13386254_3840_2160_24fps.mp4"

cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video frame.")

points = np.array([
    [284, 552],
    [3463, 610],
    [3578, 1837],
    [357, 1807]
], dtype=np.int32)

# Draw the selected quadrilateral on the original frame
cv2.polylines(
    frame,
    [points.reshape((-1, 1, 2))],
    True,
    (0, 0, 255),
    6
)

for i, (x, y) in enumerate(points, start=1):
    cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 255), -1)
    cv2.putText(
        frame,
        str(i),
        (int(x) + 12, int(y) - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 255),
        4
    )

# Resize only for display
display = cv2.resize(frame, (1280, 720))

cv2.imshow("Check Selected Pitch Points", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
