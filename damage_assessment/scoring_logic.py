from enum import Enum

class Damage(Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"

    @classmethod
    def get_color(cls, input):
        if input == Damage.RED:
            return (255,0,0)
        if input == Damage.AMBER:
            return (255,191,0)
        if input == Damage.GREEN:
            return (0,255,0)
        return (255,255,255)


def check_overlap(buildings):
    pass

def identify_walls(building, walls):
    pass

def building_scoring(walls_damage):
    pass