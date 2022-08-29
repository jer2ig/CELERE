# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, print_args, strip_optimizer)
from utils.plots import Annotator
from utils.torch_utils import select_device, smart_inference_mode
import torchvision.transforms as T
IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


@smart_inference_mode()
class Classify:
    def __init__(self,
                 weights=ROOT / 'yolov5s-cls.pt',  # model.pt path(s)
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                 imgsz=(224, 224),  # inference size (height, width)
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 **kwargs,
                 ):
        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        self.transforms = classify_transforms()
       # size = 224
       # self.transforms =  T.Compose([T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def run(self,
            weights=ROOT / 'yolov5s-cls.pt',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            nosave=False,  # do not save images/videos
            update=False,  # update all models
            project=ROOT / 'runs/predict-cls',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            **kwargs,
    ):
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=self.imgsz, transforms=classify_transforms(self.imgsz[0]))
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=self.imgsz, transforms=classify_transforms(self.imgsz[0]))
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        seen, windows, dt = 0, [], Profile()
        for path, im, im0s, vid_cap, s in dataset:
            with dt:
                pred = self.inference(im)

            # Process predictions
            for i, prob in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0 = path[i], im0s[i].copy()
                    s += f'{i}: '
                else:
                    p, im0 = path, im0s.copy()

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                s += '%gx%g ' % im.shape[-2:]  # print string
                annotator = Annotator(im0, example=str(self.names), pil=True)

                # Print results
                top5i = prob.argsort(0, descending=True)[:5].tolist()  # top 5 indices
                s += f"{', '.join(f'{self.names[j]} {prob[j]:.2f}' for j in top5i)}, "

                # Write results
                if save_img or view_img:  # Add bbox to image
                    text = '\n'.join(f'{prob[j]:.2f} {self.names[j]}' for j in top5i)
                    annotator.text((32, 32), text, txt_color=(255, 255, 255))

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
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f"{s}{dt.dt * 1E3:.1f}ms")

        # Print results
        t = dt.t / seen * 1E3  # speeds per image
        LOGGER.info(f'Speed: %.1fms per image at shape {(1, 3, *self.imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    def inference(self, im):
        im = torch.Tensor(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        results = self.model(im)
        # Post-process
        pred = F.softmax(results, dim=1)  # probabilities
        return pred

    def transform(self, img):
        return self.transforms(img)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-cls.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[224], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-cls', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    classifier = Classify(**vars(opt))
    classifier.run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
