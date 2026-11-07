from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

video_path = "VIDEOS/13386254_3840_2160_24fps.mp4"


#STORE THE BALL POSITION
ball_positions = []

frame_number  = 0 
max_frames = 100

results = model.track(
    source= video_path,
    stream= True,
    persist= True,
    classes = [32],
    conf = 0.1,
    imgsz = 648,
    verbose = False


)


for result in results:
    frame_number += 1

    frame = result.plot()

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.tolist()
        if result.boxes.id is not None:
             track_ids = result.boxes.id.int().tolist()
        else:
         track_ids = [None] * len(boxes)

        for track_id, box in zip(track_ids, boxes):

            x1, y1, x2, y2 = box

            # Calculate the centre of the ball's box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Save frame number, track ID and position
            ball_positions.append(
                (frame_number, track_id, center_x, center_y)
            )

            cv2.imshow("Ball Tracking", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Limit this first experiment to 100 frames
    if frame_number >= max_frames:
        break


cv2.destroyAllWindows()


# 4. Display a summary
print("\n===== BALL TRACKING SUMMARY =====")

print("Frames processed:", frame_number)
print("Ball positions recorded:", len(ball_positions))

if ball_positions:

    print("\nFirst 15 recorded positions:")

    for entry in ball_positions[:15]:
        print(
            "Frame:", entry[0],
            "| Ball ID:", entry[1],
            "| Position:", (entry[2], entry[3])
        )

    track_ids = sorted({
        entry[1]
        for entry in ball_positions
        if entry[1] is not None
    })

    print("\nObserved ball tracking IDs:", track_ids)

else:
    print("No ball detections were recorded.")
        
