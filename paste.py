from ultralytics import YOLO
import cv2


model = YOLO("yolo11n.pt")




results = model.predict(
    source= "VIDEOS/13386254_3840_2160_24fps.mp4",
    stream = True,
    conf=0.5,
    imgsz=640,
    verbose= False
)


for result in results:
    frame = result.plot()


    cv2.imshow("football",frame)


    if cv2.waitKey(1) == ord("q"):
       break

cv2.destroyAllWindows()
