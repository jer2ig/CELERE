"""
Annotates an image with the bounding boxes from a Yolo format text file

Usage:
    $ python annotate_image.py  fruits.jpg    --labels orange apple banana

The labels file is expected in the same location with the same name, but with the extension .txt
"""
import argparse
from pathlib import Path
import numpy as np
from utils.general import cv2, xywh2xyxy
from utils.dataloaders import img2label_paths
from utils.plots import Annotator, colors

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, default='image.jpg', help='image path')
    parser.add_argument('--labels',default = None, nargs='+', type=str, help='list of label names')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--image-size', default=None, type=int, help='size larger axis of image')
    opt = parser.parse_args()
    return opt

def main(opt):
    im0 = cv2.imread(opt.image)

    # Resize image
    if opt.image_size:
        ar = im0.shape[0] / im0.shape[1]
        if im0.shape[0] < im0.shape[1]:
            dims = (opt.image_size, int(ar*opt.image_size))
        else:
            dims = (int(opt.image_size/ar), opt.image_size)
        im0 = cv2.resize(im0, dims, interpolation=cv2.INTER_AREA)

    # Get label path and save path
    label_path = img2label_paths([opt.image])
    image_path = Path(opt.image)
    save_path = image_path.parent / (image_path.stem + "_labelled.jpg" )

    annotator = Annotator(im0, line_width=opt.line_thickness, example=str(opt.labels))
    hide_labels = not bool(opt.labels)

    # Read label file
    with open(label_path[0]) as f:
        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
        if any(len(x) > 6 for x in lb):  # is segment
            classes = np.array([x[0] for x in lb], dtype=np.float32)
            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
            lb = np.concatenate((classes.reshape(-1, 1), segments), 1)  # (cls, xywh)
        lb = np.array(lb, dtype=np.float32)

    # Annotate image
    for c, *xywh in lb:
        c = int(c)
        label = str(c) if hide_labels else opt.labels[c]
        xyxy = xywh2xyxy(np.array([xywh])).flatten()
        xyxy[0] *= im0.shape[1]
        xyxy[1] *= im0.shape[0]
        xyxy[2] *= im0.shape[1]
        xyxy[3] *= im0.shape[0]
        annotator.box_label(xyxy, label, c, color=colors(c, True))

    im0 = annotator.result()
    cv2.imwrite(save_path, im0)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
