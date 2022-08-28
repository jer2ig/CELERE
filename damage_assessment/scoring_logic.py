from enum import Enum
import collections

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
        if bbox_iou(building, w) > 0.5:
            building_walls.append(w)
    return building_walls


def building_scoring(walls_damage):
    if 1 not in walls_damage:
        return Damage.GREEN
    counts  = collections.Counter(walls_damage)
    if counts[1] > 1 or counts[1] / counts[0] > 1/4:
        return Damage.RED
    return Damage.AMBER
