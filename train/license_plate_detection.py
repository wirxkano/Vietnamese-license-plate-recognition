from train.trainer import Trainer


class LicensePlateDetectionTrainer(Trainer):
    def __init__(self, model: str, train_data: str, save_path: str):
        super().__init__(model, train_data, save_path)

    def train(self, epochs: int, imgsz: int):
        return super().train(epochs, imgsz)
