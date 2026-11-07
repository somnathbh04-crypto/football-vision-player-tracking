from ultralytics import YOLO
import cv2


# ==========================================
# 1. LOAD YOLO MODEL
# ==========================================

model = YOLO("yolo11n.pt")


# ==========================================
# 2. VIDEO + YOLO TRACKING
# ==========================================

results = model.track(
    source="VIDEOS/13386254_3840_2160_24fps.mp4",
    stream=True,
    conf=0.2,
    imgsz=448,
    verbose=False
)


# ==========================================
# 3. STORE PLAYER POSITIONS
# ==========================================

player_positions = {}

frame_number = 0
max_frames = 100


# ==========================================
# 4. PROCESS EACH VIDEO FRAME
# ==========================================

for result in results:

    frame_number += 1

    # Get the current video frame
    frame = result.orig_img.copy()

    print("Frame:", frame_number)


    # ======================================
    # 5. CHECK FOR TRACKING IDs
    # ======================================

    if result.boxes.id is not None:

        track_ids = result.boxes.id.int().tolist()
        boxes = result.boxes.xyxy.tolist()


        # ==================================
        # 6. PROCESS EACH PLAYER
        # ==================================

        for track_id, box in zip(track_ids, boxes):

            # Bounding-box coordinates
            x1, y1, x2, y2 = box


            # ------------------------------
            # Calculate player center
            # ------------------------------

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            position = (center_x, center_y)


            # ------------------------------
            # Create list for new player
            # ------------------------------

            if track_id not in player_positions:

                player_positions[track_id] = []


            # ------------------------------
            # Store player's position
            # ------------------------------

            player_positions[track_id].append(position)


            # ------------------------------
            # Draw bounding box
            # ------------------------------

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )


            # ------------------------------
            # Draw player ID
            # ------------------------------

            cv2.putText(
                frame,
                f"Player {track_id}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # ------------------------------
            # Draw current center point
            # ------------------------------

            cv2.circle(
                frame,
                (int(center_x), int(center_y)),
                5,
                (0, 0, 255),
                -1
            )


    # ======================================
    # 7. DRAW PLAYER TRAJECTORIES
    # ======================================

    for player_id, positions in player_positions.items():

        # We only want positions up to
        # the current frame

        current_positions = positions[:frame_number]


        # Need at least 2 positions
        # to draw a line

        if len(current_positions) < 2:
            continue


        for i in range(1, len(current_positions)):

            # Previous position
            point1 = (
                int(current_positions[i - 1][0]),
                int(current_positions[i - 1][1])
            )


            # Current position
            point2 = (
                int(current_positions[i][0]),
                int(current_positions[i][1])
            )


            # Draw movement trail
            cv2.line(
                frame,
                point1,
                point2,
                (0, 0, 255),
                3
            )


    # ======================================
    # 8. DISPLAY VIDEO
    # ======================================

    cv2.imshow(
        "Football Player Tracking",
        frame
    )


    # ======================================
    # 9. STOP CONDITIONS
    # ======================================

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


    if frame_number >= max_frames:
        break


# ==========================================
# 10. CLOSE WINDOWS
# ==========================================

cv2.destroyAllWindows()


# ==========================================
# 11. PRINT FINAL TRAJECTORY DATA
# ==========================================

print("\n==============================")
print("FINAL TRAJECTORY DATA")
print("==============================")

for player_id, positions in player_positions.items():

    print(
        "Player:",
        player_id,
        "| Number of positions:",
        len(positions)
    )

    if len(positions) > 0:

        print(
            "First position:",
            positions[0]
        )

        print(
            "Last position:",
            positions[-1]
        )

    print()


# ==========================================
# 12. CALCULATE DISTANCE COVERED
# ==========================================

import math

print("\n =============================")
print("DISTANCE COVERED")
print("================================")


for player_id,positions in player_positions.items():
    total_distance =0

    for i in range(1, len(positions)):
        x1,y1 = positions[i-1]
        x2,y2 = positions[i]

        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)

        total_distance += distance

        print(
            "player:",
            player_id,
            "total distance covered",
            round(total_distance, 2),
            "pixels"
        )



#============================================
# SPEED CALCULATIONS 
# ============================================

FPS = 30

print("/n =============================")
print("SPEED CALCULATION")
print("/n ==============================")



for player_id, positions in player_positions.items():

  max_speed = 0

  for i in range(1, len(position)):
       x1,y1 = positions[i-1]
       x2,y2 = positions[i]

       distance = ((x2-x1)**2+(y2-y1)**2)**0.5

       time = 1/FPS

       speed = distance/time

       if speed>max_speed:
           max_speed = speed


           print(
               "PLAYER:",
               player_id,
               "MAXIMUM SPEED",
               round(max_speed,2),
               "pixels/speed"
           )


# ==========================================
# 14. DETECT HIGH-SPEED RUNS
# ==========================================

FPS = 30

SPRINT_THRESHOLD = 500

print("\n===================================")
print("DETECT HIGH-SPEED RUNS")
print("=====================================")


for player_id,positions in player_positions.items():
     sprint_count  = 0
     in_sprint = False

     for i in range(1, len(positions)):
         x1,y1 = positions[i-1]
         x2,y2 = positions[i]
         
         distance = ((x2-x1)**2+(y2-y1)**2)**0.5
         
         time = 1/FPS
         
         speed = distance/time


         if speed>SPRINT_THRESHOLD:

             if not in_sprint:
                 sprint_count += 1
                 in_sprint = True


             else:

                 in_sprint = False


                 print(
                     "PLAYER:",
                     player_id,
                     "MAX-SPRINT COUNT",
                     sprint_count

                 )
             




         
