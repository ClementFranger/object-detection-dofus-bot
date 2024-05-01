import os

from ultralytics import YOLO
from comet_ml.api import API
from ultralytics.data.loaders import LoadScreenshots


def infer(model, dofus):
    model = YOLO(os.path.join(model['path'], 'best.pt'))

    loader = LoadScreenshots(f"screen 2 {dofus.left} {dofus.top} {dofus.width} {dofus.height}", imgsz=(1920, 1080))  # Do not forget 'screen' as source
    for screen, img, _, _ in loader:
        model.predict(img[0], show=True, imgsz=(1920, 1080))
