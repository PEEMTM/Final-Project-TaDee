from ultralytics import YOLO
# Export the model
model = YOLO(r'runs\detect\train5\weights\best.pt')  # load a custom model
model.export(format='tflite') # export the model to tflite format