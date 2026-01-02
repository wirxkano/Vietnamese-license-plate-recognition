from ultralytics import YOLO


class Trainer:
    def __init__(self, model: str, train_data: str, save_path: str):
        self.model = model
        self.train_data = train_data
        self.save_path = save_path

        self.yolo_model = None

    def train(self, epochs: int, imgsz: int):
        self.yolo_model = YOLO(self.model)
        self.yolo_model.train(
            data=self.train_data,
            epochs=epochs,
            imgsz=imgsz,
            batch=0.70,
            project=self.save_path,
        )

    def get(self):
        pass
