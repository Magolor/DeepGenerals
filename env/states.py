import sys
import gym
from utils import *

from env.const import C
from env.ufs import UnionFindSet

class GridMap(object):
    def __init__(self, W, H):
        self.W = W; self.H = H
        self.map = np.ones((W,H),dtype=np.int)
        self.capitals = []

    def data(self):
        return self.map.copy(), list(self.capitals)

    def AddCapital(self, x, y):
        self.map[x][y] = C.LAND_CAPITAL
        self.capitals.append((x,y))

    def AddCity(self, x, y):
        self.map[x][y] = C.LAND_CITY

    def AddMountain(self, x, y):
        self.map[x][y] = C.LAND_MOUNTAIN

    def connected(self):
        S = UnionFindSet([(i,j) for j in range(self.H) for i in range(self.W)])
        for i in range(self.W):
            for j in range(self.H):
                if self.map[i][j]!=C.LAND_MOUNTAIN:
                    for t in C.MOVEABLE_DIRECTIONS:
                        x,y = i+t[0],j+t[1]
                        if (x,y) in S.fa.keys() and self.map[x][y]!=C.LAND_MOUNTAIN:
                            S.union((i,j),(x,y))
        assert(len(self.capitals) > 0)
        F = S.getfa(self.capitals[0])
        for capital in self.capitals:
            if S.getfa(capital)!=F:
                return False
        return True

def NewRandomMap(W, H, num_players=2, p_mountain=0.2, p_city=0.05):
    while True:
        M = GridMap(W,H); R = [(i,j) for j in range(H) for i in range(W)]; np.random.shuffle(R)
        for i in range(num_players):
            M.AddCapital(R[i][0],R[i][1])
        R = R[num_players:]
        for i in range(int(W*H*p_mountain)):
            M.AddMountain(R[i][0],R[i][1])
        R = R[int(W*H*p_mountain):]
        for i in range(int(W*H*p_city)):
            M.AddCity(R[i][0],R[i][1])
        if M.connected():
            break
    return M

class PlayerState(object):
    def __init__(self, board_grd, board_ctr, board_arm, board_obs, num_players, turn, armies, dead=False):
        self.board_shape = board_grd.shape; assert(board_obs.shape==self.board_shape); assert(board_ctr.shape==self.board_shape); assert(board_arm.shape==self.board_shape)
        self.grd = board_grd; self.obs = board_obs; self.ctr = board_ctr; self.arm = board_arm; self.num_players = num_players; self.turn = turn; self.armies = armies; self.dead=dead
    
    def copy(self):
        return PlayerState(self.grd.copy(),self.obs.copy(),self.ctr.copy(),self.arm.copy(),self.num_players,self.turn,self.armies,self.dead)
    
    def serialize(self):
        # customize !!!
        map_data = []; A = float(sum(self.armies))
        for grd_type in range(5):
            map_data.append(torch.eq(torch.tensor(self.grd),grd_type).float())
        for obs_type in range(3):
            map_data.append(torch.eq(torch.tensor(self.obs),obs_type).float())
        map_data.append(torch.tensor(self.arm).float()/A)
        for player in range(self.num_players+2):
            map_data.append(torch.eq(torch.tensor(self.ctr),player).float())
        # stat_data = []
        # stat_data.append(torch.tensor([self.turn%50]).float())
        # stat_data.append(torch.tensor([self.armies[0]]).float())
        return torch.stack(map_data,dim=0).float()
    
    def Score(self):
        return 0

    def CityControlled(self):
        return sum([(C.HAS_HOUSE(self.grd[i][j]) and self.ctr[i][j]==C.BOARD_SELF) for i in range(self.board_shape[0]) for j in range(self.board_shape[1])])

    def CityObserved(self):
        return sum([(C.HAS_HOUSE(self.grd[i][j]) and self.obs[i][j]!=C.UNOBSERVED) for i in range(self.board_shape[0]) for j in range(self.board_shape[1])])
    
    def CityObserving(self):
        return sum([(C.HAS_HOUSE(self.grd[i][j]) and self.obs[i][j]==C.OBSERVING) for i in range(self.board_shape[0]) for j in range(self.board_shape[1])])
    
    def LandControlled(self):
        return sum([(self.ctr[i][j]==C.BOARD_SELF) for i in range(self.board_shape[0]) for j in range(self.board_shape[1])])
    
    def LandObserved(self):
        return sum([(self.obs[i][j]!=C.UNOBSERVED) for i in range(self.board_shape[0]) for j in range(self.board_shape[1])])
    
    def LandObserving(self):
        return sum([(self.obs[i][j]==C.OBSERVING) for i in range(self.board_shape[0]) for j in range(self.board_shape[1])])
    
    def CapitalObserved(self):
        return sum([(self.grd[i][j]==C.LAND_CAPITAL and self.obs[i][j]!=C.UNOBSERVED) for i in range(self.board_shape[0]) for j in range(self.board_shape[1])])
        
    def CapitalObserving(self):
        return sum([(self.grd[i][j]==C.LAND_CAPITAL and self.obs[i][j]==C.OBSERVING) for i in range(self.board_shape[0]) for j in range(self.board_shape[1])])
    
    def ArmyControlled(self):
        return sum([self.arm[i][j] for i in range(self.board_shape[0]) for j in range(self.board_shape[1]) if self.ctr[i][j]==C.BOARD_SELF])

class PlayerAction(object):
    def __init__(self, src, dir, half=False):
        assert(tuple(dir) in C.MOVEABLE_DIRECTIONS)
        self.src = src; self.dir = dir; self.dst = (src[0]+dir[0],src[1]+dir[1]); self.half = half
    
    def IsAvailableIn(self, state, player_id=0):
        return (
            (0<=self.dst[0]<state.board_shape[0] and 0<=self.dst[1]<state.board_shape[1])       # in the board
        and (state.ctr[self.src[0]][self.src[1]]==player_id+C.BOARD_SELF)                       # controlled by oneself
        and (state.grd[self.dst[0]][self.dst[1]]!=C.LAND_MOUNTAIN)                              # not moving to mountain
        and (state.arm[self.src[0]][self.src[1]]>1)                                             # have army to move
        )
    
    def __str__(self):
        return str((self.src,self.dir,self.half))

class BoardState(object):
    def __init__(self, true_board_grd=None, true_board_ctr=None, true_board_arm=None, board_obss=None, turn=0, dead=None):
        if true_board_grd is not None:
            self.board_shape = true_board_grd.shape; assert(true_board_ctr.shape==self.board_shape); assert(true_board_arm.shape==self.board_shape)
            self.num_players = len(board_obss); self.grd = true_board_grd; self.ctr = true_board_ctr; self.arm = true_board_arm; self.obss = board_obss; self.turn = turn
            self.dead = list() if dead is None else dead
    
    def serialize(self):
        return (self.grd.copy(),self.ctr.copy(),self.arm.copy(),self.obss.copy(),self.turn,list(self.dead))

    def unserialize(self, board_state):
        self = BoardState(*board_state); return self

    def copy(self):
        return BoardState().unserialize(self.serialize())

    def GetPlayerState(self, player_id):
        assert(0<=player_id<self.num_players); grd,ctr,arm,obs = self.grd.copy(),self.ctr.copy(),self.arm.copy(),self.obss[player_id].copy()
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1]):
                if ctr[i][j]==player_id+C.BOARD_SELF:  # swap to first player
                    ctr[i][j]=C.BOARD_SELF
                elif ctr[i][j]==C.BOARD_SELF:
                    ctr[i][j]=player_id+C.BOARD_SELF
        armies = [torch.sum(torch.eq(torch.tensor(ctr),i+C.BOARD_SELF) * torch.tensor(arm)) for i in range(self.num_players)]
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1]):
                if obs[i][j]==C.UNOBSERVED and grd[i][j]!=C.LAND_MOUNTAIN:    # clear unobserved status
                    grd[i][j] = C.LAND_FOG
                    ctr[i][j] = C.BOARD_FOG
                    arm[i][j] = 0
        return PlayerState(grd,ctr,arm,obs,self.num_players,self.turn,armies,player_id in self.dead)

    def GetNextState_(self, actions):
        self.turn += 1; assert(len(actions)==self.num_players)
        # Taking turns to move
        for player_id,action in enumerate(actions):
            if (player_id in self.dead) or (action is None):
                continue
            if not action.IsAvailableIn(self, player_id):
                continue
            s = action.src; e = action.dst
            army = self.arm[s[0]][s[1]]-int(np.ceil(self.arm[s[0]][s[1]]/2.) if action.half else 1)
            self.arm[s[0]][s[1]] -= army
            if self.ctr[e[0]][e[1]]==player_id+C.BOARD_SELF:        # move within ones territory
                self.arm[e[0]][e[1]] += army
            elif army > self.arm[e[0]][e[1]]:                       # win the battle, captures a land
                self.arm[e[0]][e[1]] = army-self.arm[e[0]][e[1]]
                self.ctr[e[0]][e[1]] = player_id+C.BOARD_SELF
                if self.grd[e[0]][e[1]]==C.LAND_CAPITAL:            # captures a capital
                    self.grd[e[0]][e[1]] = C.LAND_CITY
                    enemy_id = int(self.ctr[e[0]][e[1]])-C.BOARD_SELF
                    for i in range(self.board_shape[0]):
                        for j in range(self.board_shape[1]):
                            if self.ctr[i][j]==enemy_id+C.BOARD_SELF:
                                self.arm[i][j] = 1; self.ctr[i][j] = player_id+C.BOARD_SELF
                    self.dead.append(enemy_id)
            else:                                                   # lose the battle
                self.arm[e[0]][e[1]] -= army

        # Updating observation status
        for player_id,obs in enumerate(self.obss):
            for i in range(self.board_shape[0]):
                for j in range(self.board_shape[1]):
                    if obs[i][j]==C.OBSERVING:                      # observing -> observed
                        self.obss[player_id][i][j] = C.OBSERVED
            for i in range(self.board_shape[0]):
                for j in range(self.board_shape[1]):
                    if self.ctr[i][j]==player_id+C.BOARD_SELF:      # observing
                        for t in C.OBSERVABLE_DIRECTIONS:
                            if 0<=i+t[0]<self.board_shape[0] and 0<=j+t[1]<self.board_shape[1] and self.grd[i+t[0]][j+t[1]]!=C.LAND_MOUNTAIN:
                                self.obss[player_id][i+t[0]][j+t[1]] = C.OBSERVING
        
        # Generate new army
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1]):
                if self.turn%C.TURN_PER_NEW_ARMY_EMPTY==0 and self.ctr[i][j]!=C.BOARD_NEUTRAL and self.grd[i][j]==C.LAND_EMPTY:     # controlled empty
                    self.arm[i][j] += 1
                if self.turn%C.TURN_PER_NEW_ARMY_CITY==0 and self.ctr[i][j]!=C.BOARD_NEUTRAL and self.grd[i][j]==C.LAND_CITY:       # controlled city
                    self.arm[i][j] += 1
                if self.turn%C.TURN_PER_NEW_ARMY_CAPITAL==0 and self.ctr[i][j]!=C.BOARD_NEUTRAL and self.grd[i][j]==C.LAND_CAPITAL: # controlled capital
                    self.arm[i][j] += 1
        
        # True if the game ends
        return len(self.dead)>=self.num_players-1

    def GetNextState(self, moves):
        S = self.copy(); endgame = S.GetNextState_(moves); return S, endgame

class ArmyGenerator(object):
    def __init__(self, seed=None):
        self.seed = seed; np.random.seed(seed)

    def reset():
        np.random.seed(self.seed)
    
    def seed(seed=None):
        self.seed = seed; np.random.seed(seed)

class UniformArmyGenerator(ArmyGenerator):
    def __init__(self, l, r, seed=None):
        super(UniformArmyGenerator, self).__init__(seed); self.l = l; self.r = r
    
    def __call__(self):
        return np.random.randint(self.l,self.r)

def NewBoardStateFromMap(map0, army_generator=UniformArmyGenerator(25,101)):
    grd,capitals = map0.data(); num_players = int(torch.sum(torch.eq(torch.tensor(grd),C.LAND_CAPITAL))); player_id = 0
    ctr = np.ones_like(grd); arm = np.zeros_like(grd); obss = [np.zeros_like(grd) for _ in range(num_players)]
    assert(num_players==len(capitals))
    for player_id,(i,j) in enumerate(capitals):
        assert(grd[i][j]==C.LAND_CAPITAL and ctr[i][j]==1)                                      # must be capital and must be non-visited
        ctr[i][j]=player_id+C.BOARD_SELF; arm[i][j]=1; obss[player_id][i][j] = C.OBSERVING
        for t in C.OBSERVABLE_DIRECTIONS:
            if 0<=i+t[0]<grd.shape[0] and 0<=j+t[1]<grd.shape[1] and grd[i+t[0]][j+t[1]]!=C.LAND_MOUNTAIN:
                obss[player_id][i+t[0]][j+t[1]] = C.OBSERVING
    for i in range(grd.shape[0]):
        for j in range(grd.shape[1]):
            if grd[i][j]==C.LAND_CITY:
                arm[i][j] = army_generator()
    return BoardState(grd,ctr,arm,obss)
