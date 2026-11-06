from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

results = model.track(
    source = "VIDEOS/13386254_3840_2160_24fps.mp4",
    stream = True,
    conf = 0.5,
    imgsz = 448,
    verbose = False
)

for result in results:
    frame = result.plot()


    cv2.imshow("tracking",frame)

    if cv2.waitKey(200) == ord("q"):
        break

    cv2.destroyAllWindows()