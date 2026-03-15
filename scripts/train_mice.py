import os
from ultralytics import YOLO

if __name__ == "__main__":
    os.chdir(os.path.join(os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=3,
        imgsz=1280,
        batch=4,
        project="runs/detect",
        name="mice",
    )
