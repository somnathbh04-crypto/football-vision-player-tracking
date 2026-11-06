from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

fvideo = cv2.VideoCapture("VIDEOS/13386254_3840_2160_24fps.mp4")


if not fvideo.isOpened():
    print("Could not open video")
    exit()

while True:
    success, frame = fvideo.read()

    if not success:
        break


    frame = cv2.resize(frame,(1260,720)) 


    results = model(frame, conf = 0.5,verbose = False)


    annotated_frame = results[0].plot()

    #for box in result.boxes:
     #   class_id =int(box.cls)

       # if class_id == 0:
          ##  x1,x2,y1,y2 = map(int, box.xyxy[0])

           # cv2.rectangle(
               # frame,
               # (x1,y1),
               # (x2,y2),
               # (0,255,0),
               # 2
           # )


           # confidence = float(box.conf)


           # cv2.putText(
               ## frame,
               # f"PLAYER{confidence:.2f}",
               # (x1,y1 -10),
               # cv2.FONT_HERSHEY_SIMPLEX,
                #0.5,
                #(0.255,0),
              # 3 2
            #)
    cv2.imshow("football ",annotated_frame)

    cv2.namedWindow("football",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("football",1200,800)

    


    if cv2.waitKey(9000) == ord("q"):
      break

    fvideo.release()
    cv2.destroyAllWindows()