import numpy as np

class C:
    # grd
    LAND_FOG = 0
    LAND_EMPTY = 1
    LAND_MOUNTAIN = 2
    LAND_CITY = 3
    LAND_CAPITAL = 4

    # ctr
    BOARD_FOG = 0
    BOARD_NEUTRAL = 1
    BOARD_SELF = 2

    # obs
    UNOBSERVED = 0
    OBSERVED = 1
    OBSERVING = 2

    TURN_PER_NEW_ARMY_EMPTY = 25
    TURN_PER_NEW_ARMY_CITY = 1
    TURN_PER_NEW_ARMY_CAPITAL = 1

    def HAS_HOUSE(g):
        return g==C.LAND_CITY or g==C.LAND_CAPITAL

    def HAS_HOUSE_BATCH(g):
        return np.logical_or(g==C.LAND_CITY,g==C.LAND_CAPITAL)

    MOVEABLE_DIRECTIONS = ((-1,0),(0,1),(1,0),(0,-1))
    MOVEABLE_DIRECTIONS_ID = {(-1,0):0, (0,1):1, (1,0):2, (0,-1):3}
    OBSERVABLE_DIRECTIONS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1))

    # NN utils
    FEATURES = 16
    NUM_FRAME = 8
    MAX_TURN = 500
    REWARD_SCALE = 0.1
    ACTION_REWARD = 0.5
    TIME_PUNISHMENT = 1e-2