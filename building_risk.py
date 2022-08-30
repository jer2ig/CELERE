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
    sys.save_dir.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from ClassDetect import Detection
from classify.ClassPredict import Classification
from damage_assessment.scoring_logic import (Damage, identify_walls, building_scoring_cls,
                                             building_scoring_det, evaluate_wall)


def run(weights_b=ROOT / 'yolov5s.pt',  # model.pt path(s)
        weights_d=ROOT / 'yolov5s.pt',  # model.pt path(s)
        use_dam_detection=False,
        disable_walls=False,
        source=ROOT / 'data/images',  # file/dir
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/assess',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    source = str(source)

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    # Building detection model
    model_b = Detection(data=data, imgsz=imgsz, device=device, half=half, dnn=dnn, weights=weights_b)
    if use_dam_detection:
        model_d = Detection(data=data, imgsz=imgsz, device=device, half=half, dnn=dnn, weights=weights_d)
    else:
        model_d = Classification(data=data, imgsz=imgsz, device=device, half=half, dnn=dnn, weights=weights_d)

    # Dataloader
    dataset = LoadImages(source, img_size=model_b.imgsz, stride=model_b.stride, auto=model_b.pt)
    bs = 1  # batch_size

    # Run inference
    seen, windows, dt = 0, [], Profile()
    for path, im, im0s, vid_cap, s in dataset:
        with dt:
            pred = model_b.inference(agnostic_nms, augment, classes, conf_thres, im, iou_thres, max_det, path,
                                  visualize)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[-2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()   # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(model_b.names))
            crop_i = 0
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[-2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {model_b.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                buildings = []
                walls = []
                # Obtain buildings and walls
                for *xyxy, conf, cls in reversed(det):
                    cls = int(cls)
                    if model_b.names[cls] == 'house':
                        print("building")
                        buildings.append(xyxy)
                    elif model_b.names[cls] == 'wall':
                        print("wall")
                        walls.append(xyxy)
                    else:
                        print("Unknown cls: ", cls)

                for b in buildings:
                    building_walls = identify_walls(b, walls)
                    scores = []
                    if not disable_walls:
                        for w in building_walls:
                            wall = save_one_box(w, imc)
                            crop_i +=1
                            wall = model_d.transform(wall)
                            if use_dam_detection:
                                prediction = model_d.inference(agnostic_nms, augment, classes, conf_thres, wall, iou_thres, max_det, path,
                                                               visualize)
                                prediction = evaluate_wall(w, prediction, model_d)
                            else:
                                prediction = model_d.inference(wall)
                                prediction = int(torch.argmax(prediction))
                            scores.append(prediction)
                    else:
                        build = save_one_box(b, imc)
                        build = model_d.transform(build)
                        if use_dam_detection:
                            prediction = model_d.inference(agnostic_nms, augment, classes, conf_thres, build, iou_thres, max_det, path,
                                                           visualize)
                            if len(prediction[0])==0:
                                prediction = 0
                            else:
                                prediction = 2
                            print(prediction)
                        else:
                            prediction = model_d.inference(build)
                            prediction = int(torch.argmax(prediction))*2
                            print(prediction)
                        scores.append(prediction)

                    if use_dam_detection:
                        score = building_scoring_det(scores)
                    else:
                        score = building_scoring_cls(scores)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(b).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (score.value, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        label = score.value
                        annotator.box_label(b, label, color=Damage.get_color(score))

                    if save_crop:
                        save_one_box(b, imc, file=save_dir / 'crops' / score.value / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)


        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt.dt * 1E3:.1f}ms")

    # Print results
    t = dt.t / seen * 1E3 # speeds per image
    LOGGER.info(f'Speed: %.1fms per image at shape {(1, 3, *model_b.imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights_b[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-b', nargs='+', type=str, default=ROOT / 'damage_assessment/buildings.pt', help='weights for building model')
    parser.add_argument('--weights-d', nargs='+', type=str, default=ROOT / 'damage_assessment/damage_classify.pt', help='weights for damage_classification')
    parser.add_argument('--use-dam-detection', action='store_true', help='use damage detection instead of classifcation')
    parser.add_argument('--disable-walls', action='store_true', help='replace walls by buildings in damage analysis step')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/damage', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
