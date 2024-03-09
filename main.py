from ultralytics import YOLO
import torch

def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a model
    #model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO(r"runs\detect\train4\weights\best.pt")  # load a pretrained model (recommended for training)
    model.to(device)

    # Use the model
    results = model.train(data=r"E:\dataset\ObjectDetectionFinal.v1i.yolov8\data.yaml", epochs=10, imgsz=640)  # train the model

    # Remember to deactivate your virtual environment when you're done:
    # deactivate  # On Windows
      
if __name__ == '__main__':
    main()