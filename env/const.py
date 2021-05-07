@dataclass(frozen=True)
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
    PLAYER0 = 2

    # obs
    UNOBSERVED = 0
    OBSERVED = 1
    OBSERVING = 2

    TURN_PER_NEW_ARMY_EMPTY = 50
    TURN_PER_NEW_ARMY_CITY = 2
    TURN_PER_NEW_ARMY_CAPITAL = 2

    def HAS_HOUSE(g):
        return g==LAND_CITY or g==LAND_CAPITAL

    MOVEABLE_DIRECTIONS = ((-1,0),(0,1),(1,0),(0,-1))
    OBSERVABLE_DIRECTIONS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(1,1),(1,-1),(1,0),(1,1))