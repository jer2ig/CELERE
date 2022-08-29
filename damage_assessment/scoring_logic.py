from enum import Enum
import collections
import torch

from utils.metrics import bbox_iou

class Damage(Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"

    @classmethod
    def get_color(cls, input):
        if input == Damage.RED:
            return (0,0,255)
        if input == Damage.AMBER:
            return (0,191,255)
        if input == Damage.GREEN:
            return (0,255,0)
        return (255,255,255)


def identify_walls(building, walls):
    building_walls = []
    for w in walls:
        if iow(building, w) > 0.5:
            building_walls.append(w)
    return building_walls

def iow(build, wall):
    b1_x1, b1_y1, b1_x2, b1_y2 = build
    b2_x1, b2_y1, b2_x2, b2_y2 = wall
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # Union Area
    #union = w1 * h1 + w2 * h2 - inter
    wall = w2 * h2

    # IoW
    iow = inter / wall
    return iow


def building_scoring_cls(walls_damage):
    if 1 not in walls_damage:
        return Damage.GREEN
    counts  = collections.Counter(walls_damage)
    if counts[1] > 1  or counts[1] > 4*counts[0]:
        return Damage.RED
    return Damage.AMBER

def building_scoring_det(walls_damage):
    if 1 and 2 not in walls_damage:
        return Damage.GREEN
    if 2 in walls_damage:
        return Damage.RED
    counts  = collections.Counter(walls_damage)
    if counts[1] > 1 or counts[1] > 4*counts[0]:
        return Damage.RED
    return Damage.AMBER

def evaluate_wall(wall, damages, model_d):
    print(damages)
    damages=damages[0]
    if len(damages) ==0:
        return 0
    print(damages)
    print("____________")
    percent_dam = 0
    for *xyxy, conf, cls in reversed(damages):
        cls = int(cls)
        print(model_d.names[cls])
        if model_d.names[cls] == 'rebar':
            return 2
        if model_d.names[cls] != 'superficial':
            percent_dam += iow(xyxy, wall)
    if percent_dam > 0.5:
        return 2
    if percent_dam < 0.01:
        return 0
    return 1



