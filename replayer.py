import time
import pickle
import argparse
from utils import *
from env.states import BoardState, PlayerState
from env.const import C

import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtSvg import QSvgWidget
from qt_material import apply_stylesheet
from functools import partial
import numpy as np

COLORS = [QColor(0, 0, 0, 128), QColor(245, 222, 179, 128), QColor(255, 0, 0, 128), QColor(0, 0, 255, 128), QColor(0, 255, 0, 128)]
LANDSCAPES = [None,None,'mountains.png','house.png','crown.png']
EXTRA = {
    # Button colors
    'danger': '#dc3545',
    'warning': '#ff8c00',
    'success': '#228b22',
    # Font
    'font_family': 'Consolas',
    'font_size': 'mono',
    'line_height': '14px',
}

class QTimerWithPause(QTimer):
    def __init__(self, parent = None):
        QTimer.__init__(self, parent)
        self.startTime = 0
        self.interval = 0
    def startTimer(self, interval):
        self.interval = interval
        if not self.isActive():
            self.startTime = time.time()
            self.start(interval)
    def pauseTimer(self):
        if self.isActive():
            self.stop()
            elapsedTime = self.startTime - time.time()
            self.startTime -= elapsedTime
            self.interval -= int(elapsedTime*1000)

class GUI(QWidget):
    def __init__(self, replay, framerate=2):
        super(GUI, self).__init__()
        assert(len(replay)>1)
        self.num_turns = len(replay)-1
        self.shape = replay[0].board_shape
        self.W, self.H = (1280, 720)
        self.b = int(min(self.W*0.775/(self.shape[0]+1),self.H*0.950/(self.shape[1]+1)))
        self.font_size = self.b//8
        self.replay = replay
        self.active = False
        self.turn = 0

        self.font = QFont('Consolas',self.font_size); self.font.setBold(True)

        next = QPushButton("NEXT",self); next.setProperty('class', 'default'); next.clicked.connect(self.NEXT)
        next.move(int(self.W*0.85),int(self.H*0.05)); next.resize(int(self.W*0.1),int(self.H*0.4))
        next_sc = QShortcut(QKeySequence('Right'), self); next_sc.activated.connect(self.NEXT)
        prev = QPushButton("PREV",self); prev.setProperty('class', 'default'); prev.clicked.connect(self.PREV)
        prev.move(int(self.W*0.85),int(self.H*0.15)); prev.resize(int(self.W*0.1),int(self.H*0.4))
        prev_sc = QShortcut(QKeySequence( 'Left'), self); prev_sc.activated.connect(self.PREV)

        exit = QPushButton("START",self); exit.setProperty('class', 'success'); exit.clicked.connect(self.START)
        exit.move(int(self.W*0.85),int(self.H*0.25)); exit.resize(int(self.W*0.1),int(self.H*0.4))
        exit = QPushButton("PAUSE",self); exit.setProperty('class', 'warning'); exit.clicked.connect(self.PAUSE)
        exit.move(int(self.W*0.85),int(self.H*0.35)); exit.resize(int(self.W*0.1),int(self.H*0.4));
        next_sc = QShortcut(QKeySequence('Space'), self); next_sc.activated.connect(self.SWITCH)

        self.num_players = 2; self.current_player = 0
        for player in range(self.num_players+1):
            p = QPushButton("PLAYER %d"%player if player else "GOD",self); p.setProperty('class', 'default' if player else 'warning'); p.clicked.connect(partial(self.PLAYER,player))
            p.move(int(self.W*0.85),int(self.H*(0.45+player*0.1))); p.resize(int(self.W*0.1),int(self.H*0.4))
            player_sc = QShortcut(QKeySequence('%d'%player), self); player_sc.activated.connect(partial(self.PLAYER,player))

        exit = QPushButton("EXIT",self); exit.setProperty('class', 'danger'); exit.clicked.connect(self.EXIT)
        exit.move(int(self.W*0.85),int(self.H*0.85)); exit.resize(int(self.W*0.1),int(self.H*0.4))
        exit_sc = QShortcut(QKeySequence('Escape'), self); exit_sc.activated.connect(self.EXIT)
        
        pgup_sc = QShortcut(QKeySequence('PgUp'), self); pgup_sc.activated.connect(self.BEGIN)
        pgdn_sc = QShortcut(QKeySequence('PgDown'), self); pgdn_sc.activated.connect(self.END)

        self.framerate = framerate
        self.timer = QTimerWithPause(self)
        self.timer.timeout.connect(self.NEXT)
        self.InitUI()

    def Board(self):
        return self.replay[self.turn].GetPlayerState(self.current_player-1) if self.current_player else self.replay[self.turn]

    def PosToPix(self, x, y):
        return self.b+x*self.b, self.b+y*self.b

    def PixToPos(self, x, y):
        return int((x-self.b)/self.b), int((y-self.b)/self.b)

    def PLAYER(self, player):
        self.current_player = player
        self.update()
        self.repaint()

    def START(self):
        self.active = True
        self.timer.startTimer(1000//self.framerate)

    def PAUSE(self):
        self.active = False
        self.timer.pauseTimer()

    def SWITCH(self):
        if self.active:
            self.PAUSE()
        else:
            self.START()

    def EXIT(self):
        QCoreApplication.instance().quit()

    def NEXT(self):
        if self.turn < self.num_turns:
            self.turn += 1;
            self.update()
            self.repaint()

    def PREV(self):
        if self.turn > 0:
            self.turn -= 1;
            self.update()
            self.repaint()

    def BEGIN(self):
        self.PAUSE()
        if self.turn != 0:
            self.turn = 0
            self.update()
            self.repaint()

    def END(self):
        self.PAUSE()
        if self.turn != self.num_turns:
            self.turn = self.num_turns
            self.update()
            self.repaint()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        qp.setPen(QPen(Qt.gray, 3))
        qp.drawLine(int(self.W*0.80),0,int(self.W*0.80),self.H);
        qp.setPen(QPen(Qt.black, 3))
        qp.setFont(self.font)
        for i in range(self.shape[0]):
            x,_ = self.PosToPix(i,0); text = "%2d"%i; qp.drawText(int(x+self.b/2.-self.font_size*len(text)/2)-6,int(self.b/2.+self.font_size/2),text)
        for j in range(self.shape[1]):
            _,y = self.PosToPix(0,j); text = "%2d"%j; qp.drawText(int(self.b/2.-self.font_size*len(text)/2),int(y+self.b/2.+self.font_size/2),text)
        board = self.Board()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x,y = self.PosToPix(i,j)

                color = board.ctr[i][j]
                if self.current_player:
                    if color==2:
                        color = self.current_player+1
                    elif color==self.current_player+1:
                        color = 2
                if isinstance(board,PlayerState) and board.obs[i][j]!=C.OBSERVING:
                    color = 0
                qp.setBrush(COLORS[color])
                qp.drawRect(x,y,self.b,self.b)

                landscape = LANDSCAPES[board.grd[i][j]]
                if landscape is not None:
                    icon = QImage('assets/'+landscape)
                    targ = QRectF(x+self.b*0.2,y+self.b*0.2,self.b*0.6,self.b*0.6)
                    qp.drawImage(targ,icon)

                if board.arm[i][j]>0:
                    self.font.setBold( True); qp.setFont(self.font); qp.setPen(QPen(Qt.black))
                    text = "%d"%board.arm[i][j]; qp.drawText(int(x+self.b/2.-self.font_size*len(text)/2),int(y+self.b/2.+self.font_size/2)+1,text)
                    text = "%d"%board.arm[i][j]; qp.drawText(int(x+self.b/2.-self.font_size*len(text)/2),int(y+self.b/2.+self.font_size/2)-1,text)
                    text = "%d"%board.arm[i][j]; qp.drawText(int(x+self.b/2.-self.font_size*len(text)/2)+1,int(y+self.b/2.+self.font_size/2),text)
                    text = "%d"%board.arm[i][j]; qp.drawText(int(x+self.b/2.-self.font_size*len(text)/2)-1,int(y+self.b/2.+self.font_size/2),text)
                    self.font.setBold(False); qp.setFont(self.font); qp.setPen(QPen(Qt.white))
                    text = "%d"%board.arm[i][j]; qp.drawText(int(x+self.b/2.-self.font_size*len(text)/2),int(y+self.b/2.+self.font_size/2),text)
                    qp.setPen(QPen(Qt.black, 3))
        qp.setFont(QFont('Consolas',12))
        text = "Turn %4d/%4d"%(self.turn,self.num_turns); qp.drawText(int(self.W*0.84),int(self.H*0.80),text)
        qp.end()

    # def mousePressEvent(self, e):
        # print("click",e.x(),e.y())
        # QCoreApplication.instance().quit()
        # if self.win:
        #     return
        # pos = self.CoordinateTopos(e.x(), e.y())
        # # QMessageBox.information(self, '框名', "(%d,%d)" %(pos[0],pos[1]))
        # if (pos[0] < 0) or (pos[0] >= self.S) or (pos[1] < 0) or (pos[1] >= self.S) or (
        #         pos not in self.Board.getAvailableActions()):
        #     return
        # else:
        #     self.down(pos)
        #     self.update()
        #     self.repaint()
        #     self.run()

    def InitUI(self):
        self.resize(self.W, self.H)
        self.setWindowTitle('Generals.io Simluator')
        self.show()

def Replay(replay_id, offset = C.NUM_FRAME-1, framerate = 10):
    if ".replay" not in replay_id:
        replay_path = "replays/"+replay_id+".replay"
    else:
        replay_path = "replays/"+replay_id
    replay_file = open(replay_path,"rb")
    replay = [BoardState(*h) for h in pickle.load(replay_file)][offset:]
    app = QApplication(sys.argv); apply_stylesheet(app, theme='light_blue.xml', extra=EXTRA)
    gui = GUI(replay, framerate = framerate); app.exec_()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="id", help="Replay ID", type=str, default="PPO_First_2021-06-22[12.31.59]_KDI6N0AM")
    parser.add_argument("-f", dest="framerate", help="Turn per Sec", type=int, default=20)
    args = parser.parse_args()
    Replay(args.id, framerate=args.framerate)