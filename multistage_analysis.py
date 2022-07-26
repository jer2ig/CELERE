"""
Run inference on a picture to determine the building risk score

Usage - sources:
    $ python path/to/detect.py  --weights det_components.pt     # Trained YOLO weights to identify building components
                                --weights_columns columns.pt    # Trained Resnet weights to classify concrete columns
                                --source img.jpg            # image
                                        path/               # directory
                                        path/*.jpg          # glob

"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (print_args, check_img_size, non_max_suppression, cv2, LOGGER, xyxy2xywh, scale_coords,
                           colorstr, increment_path)
from detect import run
from utils.torch_utils import select_device, time_sync
from utils.plots import save_one_box, Annotator, colors
from damage_assessment.scoring_logic import building_scoring


@torch.no_grad()
def run(
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        weights_columns = ROOT / 'yolov5s.pt',  # model.pt path(s)
        project=ROOT / 'runs/analyse',  # save results to project/name
        name='exp',  # save results to project/name
        nosave=False,  # do not save images/videos
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        imgsz=(640, 640),  # inference size (height, width)
        save_txt=True, # save results to text file
        save_conf=True,  # save confidences in --save-txt labels
        line_thickness=3,  # bounding box thickness (pixels)
):
    source = str(source)
    save_img = not nosave  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    device = select_device(device)

    # Load building analysis model
    model_building = DetectMultiBackend(weights, device=device)
    stride, names, pt = model_building.stride, model_building.names, model_building.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load damage detection model
    #model_damage = DetectMultiBackend(weights, device=device)
    model_damage = Classify(weights, device=device)


    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    # Run inference
    model_building.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    model_damage.warmup(imgsz=(1, 3, *imgsz))
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred_buildings = model_building(im, augment=True)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        classes = None # Filter by classes
        pred_buildings = non_max_suppression(pred_buildings, classes=classes)
        dt[2] += time_sync() - t3
        print("Predictions after NMS - buildings")
        print(pred)


        # Inference
        pred_damage = model_damage(im, augment=True)
        print("Predictions after inference - damage")
        print(pred_buildings)

        # NMS
        classes = None # Filter by classes
        pred_damage = non_max_suppression(pred, classes=classes)
        dt[2] += time_sync() - t3
        print("Predictions after NMS")
        print(pred_damage)
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        save_crop = True
        for i, det_build, det_dam in enumerate(pred_buildings, pred_damage):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            save_detections = save_dir
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det_build):
                # Rescale boxes from img_size to im0 size
                det_build[:, :4] = scale_coords(im.shape[2:], det_build[:, :4], im0.shape).round()
                det_dam[:, :4] = scale_coords(im.shape[2:], det_dam[:, :4], im0.shape).round()

                # Print results
                for c in det_build[:, -1].unique():
                    n = (det_build[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det_build):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop:  # Add bbox to image
                        c = int(cls)  # integer class
                        label =  f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, c,  color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            risk = building_scoring(det_build, det_dam)

            # Stream results
            im0 = annotator.result()

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)


        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='Component detection weights')
    parser.add_argument('--weights_columns', type=str, default=ROOT / 'yolov5s.pt', help='Component detection weights')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/glob')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
