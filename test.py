import os
import cv2
import argparse
import matplotlib.pyplot as plt

from ultralytics import YOLO
from paddleocr import PaddleOCR

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Folder to input images")
parser.add_argument("--lp_detector", required=True, help="Checkpoint path")
args = parser.parse_args()

lp_detector = YOLO(args.lp_detector)
letter_detector = PaddleOCR(lang="en")

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
        res = letter_detector.predict(img)
        chars = res[0]["rec_texts"]
        if len(chars) == 2:  # 2 line plates
            txt = chars[0] + chars[1]
        else:
            txt = chars[0]

        cv2.putText(img, txt, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        # print(res[0]["rec_scores"])

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
            res = letter_detector.predict(crop_img)
            chars = res[0]["rec_texts"]
            if len(chars) == 2:  # 2 line plates
                txt = chars[0] + "-" + chars[1]
            else:
                txt = chars[0]

            cv2.putText(
                img,
                txt,
                (int(xmin), int(ymin - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )
            # print(res[0]["rec_scores"])

    plt.imshow(img)
    plt.axis("off")
    plt.show()
