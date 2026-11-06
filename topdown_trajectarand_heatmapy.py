from ultralytics import YOLO
import cv2
import numpy as np


video_path = "VIDEOS/13386254_3840_2160_24fps.mp4"
max_frame = 100
output_path = "VIDEOS/topdown_trajectory.jpg"


model = YOLO("yolo11n.pt")

src_points = np.float32([
     (691, 560),    #top-left
     (3475, 602),   #top-right
     (3532, 1814),  #bottom-right
     (280, 1741)   #bottom-left
])


distance_points = np.float32([
    [0,0],
    [999,0],
    [999,599],
    [0,599]
])



matrix  = cv2.getPerspectiveTransform(
    src_points,
    distance_points
)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("could not open video")

success, first_frame = cap.read()
cap.release()


if not success:
    raise RuntimeError("could not read first frame ")


top_down_background= cv2.warpPerspective(
    first_frame,
    matrix,
    (1000,600)
)

#TO STORE PLAYER POSITIONS
player_position = {}
frame_number = 0


#NOW TRACK PLAYERS THROUGH THE VIDEO

results = model.track(
    source=video_path,
    stream=True,
    persist=True,
    conf=0.2,
    imgsz=448,
    verbose=False
)


for result in results:
    frame_number += 1
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
            
            
                        top_x = float(
                            transformed_point[0][0][0]
                        )
            
                        top_y = float(
                            transformed_point[0][0][1]
                        )

                        if frame_number <= 3:
                             print(
                                  "FRAME:",frame_number,
                                  "ID:",track_id,
                                  (round(top_x,1),round(top_y,1))
                             )

                        if not(0 <= top_x <= 1000 and 0 <= top_y <= 600):
                               if frame_number <= 3:
                                   print("  -> Rejected: outside top-down image")
                              
                                   continue
                               print("<- accepted")

                        position = (top_x,top_y)

                        #CREATING A LIST , FOR WHEN WE SEE A TRACK_ID
                        if not track_id in player_position:
                              player_position[track_id] = []

                        player_position[track_id].append(position)#SAVING THE PLAYER POSITION


            if frame_number >= max_frame:
                        break
            topdown_image = top_down_background.copy()

for player_id, positions in player_position.items():

    # A line needs at least two positions
    if len(positions) < 2:
        continue

    for i in range(1, len(positions)):

        previous_point = (
            int(positions[i - 1][0]),
            int(positions[i - 1][1])
        )

        current_point = (
            int(positions[i][0]),
            int(positions[i][1])
        )

        cv2.line(
            topdown_image,
            previous_point,
            current_point,
            (0, 0, 255),
            2
        )

    # Mark the final recorded position
    last_x = int(positions[-1][0])
    last_y = int(positions[-1][1])

    cv2.circle(
        topdown_image,
        (last_x, last_y),
        5,
        (0, 255, 255),
        -1
    )

    cv2.putText(
        topdown_image,
        f"ID {player_id}",
        (last_x + 7, last_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )


# ==========================================
# 9. SAVE AND DISPLAY ONE IMAGE
# ==========================================

saved = cv2.imwrite(output_path, topdown_image)

if not saved:
    raise RuntimeError("Could not save the output image.")

print("\nFrames processed:", frame_number)
print("Tracked player IDs:", len(player_position))

for player_id, positions in player_position.items():
    print(f"Player {player_id}: {len(positions)} recorded positions")

print("\nSaved image:", output_path)

cv2.namedWindow("Top-down Trajectories", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Top-down Trajectories", 1000, 600)
cv2.imshow("Top-down Trajectories", topdown_image)

#################################
#CREATING HEAT MAPS
#################################

#AT FIRST CREATE A EMPTY HEAT MAP, AS THE SIZE OF THE TOP-DOWN IMAGE
# ==========================================
# CREATE PLAYER HEATMAP
# ==========================================

# 1. Create an empty heatmap
heatmap = np.zeros((600, 1000), dtype=np.float32)

valid_positions = 0

# 2. Count each stored player position
for positions in player_position.values():

    for x, y in positions:

        x = int(round(x))
        y = int(round(y))

        # Stay inside the image boundaries
        if 0 <= x < 1000 and 0 <= y < 600:

            heatmap[y, x] += 1
            valid_positions += 1


print("Valid positions used for heatmap:", valid_positions)
print("Maximum position count:", heatmap.max())


# 3. Check that we have data
if valid_positions == 0:

    print("No valid positions available for the heatmap.")

else:

    # 4. Smooth the position counts
    heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=18,
        sigmaY=18
    )

    # 5. Normalize the heatmap
    heatmap_normalized = cv2.normalize(
        heatmap,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 6. Convert intensity values into colours
    heatmap_color = cv2.applyColorMap(
        heatmap_normalized,
        cv2.COLORMAP_JET
    )

    # 7. Prepare the original top-down pitch
    background = top_down_background.copy()

    # 8. Blend the coloured heatmap with the pitch
    blended = cv2.addWeighted(
        background,
        0.35,
        heatmap_color,
        0.65,
        0
    )

    # Apply the colour overlay only where the heat is visible
    mask = heatmap_normalized > 10

    background[mask] = blended[mask]

    # 9. Save the heatmap overlay
    output_path = "VIDEOS/player_heatmap.jpg"

    saved = cv2.imwrite(output_path, background)

    if not saved:
        raise RuntimeError("Could not save the heatmap.")

    print("Heatmap saved:", output_path)

    # 10. Display the heatmap overlay
    cv2.namedWindow("Player Heatmap", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Player Heatmap", 1000, 600)

    cv2.imshow("Player Heatmap", background)

    # Also display the coloured heatmap by itself
    cv2.imshow("Heatmap Colours Only", heatmap_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()                       





