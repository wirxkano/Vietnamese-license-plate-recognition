import argparse

from train.letter_recognition import LetterRecognitionTrainer
from train.license_plate_detection import LicensePlateDetectionTrainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_path_1",
    required=True,
    help="Path to save checkpoint (license plate detection)",
)
parser.add_argument(
    "--save_path_2", required=True, help="Path to save checkpoint (letter regconition)"
)

args = parser.parse_args()

detector = LicensePlateDetectionTrainer(
    model="yolo11n.pt",
    train_data="./dataset/license_plate.yaml",
    save_path=args.save_path_1,
)

regconizor = LetterRecognitionTrainer(
    model="yolo11n.pt",
    train_data="./dataset/letter.yaml",
    save_path=args.save_path_2,
)

detector.train(epochs=30, imgsz=320)
regconizor.train(epochs=30, imgsz=640)
