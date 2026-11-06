from ultralytics import YOLO


#loading a pretrained model
model = YOLO("yolo11n.pt")

#detect object in a image
resultant = model("images/football.webp")

#display of resultant
resultant[0].show()
