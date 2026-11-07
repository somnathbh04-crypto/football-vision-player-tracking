from ultralytics import YOLO
import cv2
import math

# 1. Load the model
model = YOLO("yolo11n.pt")

video_path = "VIDEOS/13386254_3840_2160_24fps.mp4"

max_frames = 200
frame_number = 0

ball_detected_frames = 0
possession_counts = {}

# 2. Detect and track both people and the ball
results = model.track(
    source=video_path,
    stream=True,
    persist=True,
    classes=[0, 32],  # 0 = person, 32 = sports ball
    conf=0.15,
    imgsz=640,
    verbose=False
)

# 3. Process each frame
for result in results:

    frame_number += 1
    frame = result.plot()

    players = []
    balls = []

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.int().tolist()
        confidences = result.boxes.conf.tolist()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().tolist()
        else:
            track_ids = [None] * len(boxes)

        # 4. Separate player detections from ball detections
        for track_id, box, class_id, confidence in zip(
            track_ids, boxes, classes, confidences
        ):

            x1, y1, x2, y2 = box

            if class_id == 0:
                # Estimate where the player's feet touch the ground
                foot_x = (x1 + x2) / 2
                foot_y = y2

                players.append({
                    "id": track_id,
                    "foot": (foot_x, foot_y),
                    "height": y2 - y1
                })

            elif class_id == 32:
                # Centre of the ball bounding box
                ball_x = (x1 + x2) / 2
                ball_y = (y1 + y2) / 2

                balls.append({
                    "id": track_id,
                    "center": (ball_x, ball_y),
                    "confidence": confidence
                })

    # 5. Estimate the nearest player to the ball
    if balls:

        ball_detected_frames += 1

        # If several ball candidates appear, choose the
        # one with the highest detection confidence.
        ball = max(balls, key=lambda b: b["confidence"])

        ball_x, ball_y = ball["center"]

        cv2.circle(
            frame,
            (int(ball_x), int(ball_y)),
            8,
            (0, 255, 255),
            -1
        )

        closest_player = None
        closest_distance = float("inf")

        for player in players:

            foot_x, foot_y = player["foot"]

            distance = math.hypot(
                ball_x - foot_x,
                ball_y - foot_y
            )

            if distance < closest_distance:
                closest_distance = distance
                closest_player = player

        # Use a provisional distance threshold that scales
        # with the player's apparent height in the image.
        if closest_player is not None:

            threshold = max(
                35,
                0.45 * closest_player["height"]
            )

            if closest_distance <= threshold:

                player_id = closest_player["id"]

                if player_id is not None:
                    possession_counts[player_id] = (
                        possession_counts.get(player_id, 0) + 1
                    )

                foot_x, foot_y = closest_player["foot"]

                cv2.line(
                    frame,
                    (int(ball_x), int(ball_y)),
                    (int(foot_x), int(foot_y)),
                    (0, 255, 255),
                    2
                )

                if player_id is not None:
                    message = f"Possible possession: Player {player_id}"
                else:
                    message = "Nearest player; tracking ID unavailable"

            else:
                message = "No clear nearby player"

        else:
            message = "Ball detected, no players detected"

    else:
        message = "Ball not detected in this frame"

    # 6. Display the estimate on the video
    cv2.putText(
        frame,
        message,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Possession Estimation", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if frame_number >= max_frames:
        break

cv2.destroyAllWindows()

# 7. Print the results
print("\n===== POSSESSION ESTIMATION SUMMARY =====")
print("Frames processed:", frame_number)
print("Frames with ball detections:", ball_detected_frames)

print("\nPossible possession estimates by player:")

for player_id, count in sorted(possession_counts.items()):
    print(f"Player {player_id}: {count} frames")
