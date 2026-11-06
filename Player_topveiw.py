from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO("yolo11n.pt")

src_points = np.float32([
     (691, 560),    #top-left
     (3475, 602),   #top-right
     (3532, 1814),  #bottom-right
     (280, 1741)   #bottom-left
])


top_down_points = np.float32([
    [0,0],
    [1000,0],
    [1000,600],
    [0,600]
])


matrix = cv2.getPerspectiveTransform(
    src_points,
    top_down_points
)

results = model.track(
    source="VIDEOS/13386254_3840_2160_24fps.mp4",
    stream =True,

    conf =0.2,
    imgsz = 448,
    verbose = False
)

frame_number = 0
for result in results:
    frame_number += 1 


    #Creating A BLANK TOP-DOWN
    top_veiw = cv2.warpPerspective(
        result.orig_img,
        matrix,
        (1000,600)
    ) 

    #CHECK IF TRACK_ID IS EXITS OR NOT
    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().tolist()
        boxes = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.tolist()


        for track_id, box, class_id in zip(
            track_ids,
            boxes,
            classes
        ):
            #ONLY PLAYERS
            if class_id != 0:
                continue

            #ROUNDING/BOUNDING BOX
            x1,y1,x2,y2 = box


            #PLAYER CENTER
            foot_x = (x1 + x2)/2
            foot_y = y2


            point = np.array(
                [[[ foot_x,foot_y ]]],
                dtype= np.float32
            )



            transformed_point = cv2.perspectiveTransform(
                point,
                matrix
            )


            top_x = int(
                transformed_point[0][0][0]
            )

            top_y = int(
                transformed_point[0][0][1]
            )


            #DRAWING PLAYERS ON THE TOP-DOWN VEIW

            cv2.circle(
                top_veiw,
                (top_x,top_y),
                8,
                (0,0,255),
                -1
            )

            #SHOW PLAYER ID
            #cv2.putText(
            
               # top_veiw,
                #f"{top_veiw}",
                #(top_x+10,top_y),
                #v2.FONT_HERSHEY_COMPLEX,
                #0.5,
                # 1
            #)


            #SHOW THE TOP-DOWN VEIW
            cv2.imshow(
                "top-down veiw",
                top_veiw
            )


            #PRESS Q TO STOP IT
            if cv2.waitKey(100) == ord("q"):
                break


            cv2.destroyAllWindows()
