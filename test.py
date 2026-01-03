import os
import cv2
import argparse
import utils
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
    if img.endswith((".jpg", ".jpeg", ".png"))
]

results = lp_detector(imgs, imgsz=640, conf=0.25, iou=0.45, device=0)

for result in results:
    bbox_list = result.boxes.xyxy.tolist()
    if len(bbox_list) == 0:
        img = result.orig_img
        res = letter_detector.predict(img)
        chars = res[0]["rec_texts"]
        chars = [utils.normalize_plate(t) for t in chars]
        if len(chars) == 2:  # 2 lines plate
            txt = (
                chars[0] + "-" + chars[1]
                if len(chars[0]) <= len(chars[1])
                else chars[1] + "-" + chars[0]
            )
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

            flag = False
            for cc in range(0, 2):
                for ct in range(0, 2):
                    res = letter_detector.predict(utils.deskew(crop_img, cc, ct))
                    chars = res[0]["rec_texts"]
                    chars = [utils.normalize_plate(t) for t in chars]
                    print(chars)
                    if len(chars) == 0:
                        continue
                    if len(chars) == 2:  # 2 lines plate
                        txt = (
                            chars[0] + "-" + chars[1]
                            if len(chars[0]) <= len(chars[1])
                            else chars[1] + "-" + chars[0]
                        )
                    else:
                        txt = chars[0]
                    flag = True
                    break

                if flag:
                    break

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

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
