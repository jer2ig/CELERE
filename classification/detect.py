import argparse

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
from PIL import Image
import numpy as np


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Classification neural net detection script')
parser.add_argument('image', type=str, help='path to image ')
parser.add_argument('params', type=str, help='path to model parameters')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')


@torch.no_grad()
def detect(image, params, arch):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    stored_dict = torch.load(params, map_location=device)
    model = models.__dict__[stored_dict["arch"]]()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        stored_dict = stored_dict["state_dict"]
    else:
        stored_dict = {k.replace('module.',''): v for k, v in stored_dict["state_dict"].items()}

    model.load_state_dict(stored_dict)
 #   print(model)
    img = Image.open(image)


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    standardize = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
   # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
   # img = np.ascontiguousarray(img)
   # im = torch.from_numpy(img).to(device)
    im = standardize(img)
    im = torch.unsqueeze(im, dim=0)
    print(im.size())
    pred = model(im)
    cls = torch.argmax(pred)
    print(cls)
    top5_prob, top5_catid = torch.topk(pred, 5)
    print(top5_prob)
    print(top5_catid)


def main():
    args = parser.parse_args()
  #  print(**vars(args))
    detect(**vars(args))

if __name__ == '__main__':
    main()
