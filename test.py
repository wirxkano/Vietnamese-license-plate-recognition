import os
import cv2
import argparse
import utils
import matplotlib.pyplot as plt

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Folder to input images")
parser.add_argument("--lp_detector", required=True, help="Checkpoint path")
parser.add_argument("--letter_regconizor", required=True, help="Checkpoint path")
args = parser.parse_args()

lp_detector = YOLO(args.lp_detector)
letter_regconizor = YOLO(args.letter_regconizor)

imgs = [
    os.path.join(args.input, img)
    for img in os.listdir(args.input)
    if img.endswith(("jpg", "jpeg", "png"))
]

results = lp_detector(imgs)
for result in results:
    bbox_list = result.boxes.xyxy.tolist()
    if len(bbox_list) == 0:
        img = result.orig_img
        lp = utils.read_char_in_plate(letter_regconizor, img)
        if lp != "unknown":
            cv2.putText(
                img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
            )
    else:
        for box in bbox_list:
            xmin, ymin, xmax, ymax = box
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            w, h = xmax - xmin, ymax - ymin

            img = result.orig_img
            crop_img = img[ymin : ymin + h, xmin : xmin + w]
            cv2.rectangle(
                img,
                (int(xmin), int(ymin)),
                (int(xmax), int(ymax)),
                color=(0, 0, 225),
                thickness=2,
            )
            for cc in range(0, 2):
                for ct in range(0, 2):
                    lp = utils.read_char_in_plate(
                        letter_regconizor, utils.deskew(crop_img, cc, ct)
                    )
                    if lp != "unknown":
                        cv2.putText(
                            img,
                            lp,
                            (int(xmin), int(ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (36, 255, 12),
                            2,
                        )
    plt.imshow(img)
    plt.axis("off")
    plt.show()
