from ultralytics import YOLO

model = YOLO("yolo11n.pt")

result = model("images/football.webp")

result = result[0]

print("Number of detection:" , len(result.boxes))

for box in result.boxes:
    print("classes:",int(box.cls))
    print("confidence :",float(box.conf))
    print("candidates :",box.xyxy.tolist())
    print("___________________")




    result.show()
    
    

      