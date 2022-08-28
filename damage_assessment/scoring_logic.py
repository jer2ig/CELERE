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


def check_overlap(buildings):
    pass

def identify_walls(building, walls):
    building_walls = []
    for w in walls:
        print("building", building)
        print("wall", w)
        if iow(building, w) > 0.5:
            building_walls.append(w)
    return building_walls

def iow(build, wall):
    b1_x1, b1_y1, b1_x2, b1_y2 = build
    b2_x1, b2_y1, b2_x2, b2_y2 = wall
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    print(b1_x1)
    print(b1_y1)
    print(w1)
    print(h1)

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    print(inter)
    # Union Area
    #union = w1 * h1 + w2 * h2 - inter
    wall = w2 * h2

    # IoW
    iow = inter / wall
    print(iow)
    return iow


def building_scoring(walls_damage):
    if 1 not in walls_damage:
        return Damage.GREEN
    counts  = collections.Counter(walls_damage)
    if counts[1] > 1 or counts[1] / counts[0] > 1/4:
        return Damage.RED
    return Damage.AMBER
