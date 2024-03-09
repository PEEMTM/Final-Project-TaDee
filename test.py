from ultralytics import YOLO

model = YOLO(r'runs\detect\train5\weights\best.pt')  # load a custom model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg", save=True, conf=0.5)  # predict on an image