#Import All the Required Libraries
from ultralytics import YOLO

#Load the YOLO Model
model = YOLO("models/best.pt")

#Object Detection
results = model.predict(source = "input_videos/video.mp4", save=True)

#Tracking
#results = model.track(source = "input_videos/video.mp4", save=True, persist=True)
