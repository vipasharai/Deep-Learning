from ultralytics import YOLO

# configure the tracking parameters and run the tracker
model=YOLO("yolo11n.pt")
# result=model.track(source="", conf=0.5, show=True, save=True)
results = model.track(source="video.mp4", conf=0.5, show=True, save=True)
print(results)