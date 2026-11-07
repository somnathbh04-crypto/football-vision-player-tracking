from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo11n.pt")

video_path = "VIDEOS/13386254_3840_2160_24fps.mp4"


results = model.predict(
    source = video_path,
    stream = True,
    classes = [32],
    conf = 0.10,
    imgsz = 640,
    verbose = False
)


for result in results:
    frame = result.plot()

    frame = cv2.resize(frame,(1200,720))

    cv2.imshow("ball_detection",frame)

    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
