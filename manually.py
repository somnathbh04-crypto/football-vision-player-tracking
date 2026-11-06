from ultralytics import YOLO

import cv2
model = YOLO("yolo11n.pt")
result = model("images/soccer-football-championship-stadium-crowd-260nw-2247676789.webp")


image = cv2.imread("images/soccer-football-championship-stadium-crowd-260nw-2247676789.webp")


for box in result[0].boxes:
    class_id = int(box.cls)


    if class_id ==0: #person
        x1,y1,x2,y2 = map(int, box.xyxy[0])


    cv2.rectangle(
       image,
       (x1,y1),
       (x2,y2),
       (0,255,0),
        2
    )
cv2.namedWindow("players",cv2.WINDOW_NORMAL)
cv2.resizeWindow("players",1200,800)
cv2.imshow("players",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

