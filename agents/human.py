from .base import *
import time
import pickle
import argparse
from utils import *
from env.states import BoardState, PlayerState, PlayerAction
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
SELECTED = QColor(255, 255, 0, 144); COULD_SELECT = QColor(255, 255, 0, 96)
LANDSCAPES = [None,None,'mountains.png','house.png','crown.png']
PLAYER_NAME = [None, 'Red', 'Blue', 'Green']
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

class GUI(QWidget):
    def __init__(self, board, agent_id, parent):
        super(GUI, self).__init__()
        self.num_turns = board.turn
        self.shape = board.board_shape
        self.W, self.H = (1280, 720)
        self.b = int(min(self.W*0.775/(self.shape[0]+1),self.H*0.950/(self.shape[1]+1)))
        self.font_size = self.b//8
        self.turn = board.turn
        self.board = board
        self.current_player = agent_id+1
        self.state = 2
        self.selected = None
        self.parent = parent

        self.font = QFont('Consolas',self.font_size); self.font.setBold(True)

        exit = QPushButton("CHEAT",self); exit.setProperty('class', 'warning'); exit.clicked.connect(self.CHEAT)
        exit.move(int(self.W*0.85),int(self.H*0.65)); exit.resize(int(self.W*0.1),int(self.H*0.1))

        exit = QPushButton("SKIP",self); exit.setProperty('class', 'success'); exit.clicked.connect(self.SKIP)
        exit.move(int(self.W*0.85),int(self.H*0.75)); exit.resize(int(self.W*0.1),int(self.H*0.1))
        exit_sc = QShortcut(QKeySequence('Space'), self); exit_sc.activated.connect(self.SKIP)

        exit = QPushButton("EXIT",self); exit.setProperty('class', 'danger'); exit.clicked.connect(self.EXIT)
        exit.move(int(self.W*0.85),int(self.H*0.85)); exit.resize(int(self.W*0.1),int(self.H*0.1))
        exit_sc = QShortcut(QKeySequence('Escape'), self); exit_sc.activated.connect(self.EXIT)
        buttons = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x,y = self.PosToPix(i,j)
                buttons.append(QPushButton("CLICK(%d,%d)"%(i,j), self))
                transp = QGraphicsOpacityEffect()
                transp.setOpacity(0.0)
                buttons[-1].setGraphicsEffect(transp)
                buttons[-1].setGeometry(x,y,self.b,self.b)
                buttons[-1].clicked.connect(partial(self.SELECT,i,j))
        
        self.InitUI()

    def Update(self, board):
        print("Update Called.")
        self.board = board
        self.state = 0
        self.update()
        self.repaint()
        while self.state != 2:
            self.parent.app.processEvents()
            time.sleep(0.01)
        return self.action

    def Board(self):
        return self.board.GetPlayerState(self.current_player-1) if self.current_player else self.board

    def PosToPix(self, x, y):
        return self.b+x*self.b, self.b+y*self.b

    def PixToPos(self, x, y):
        return int((x-self.b)/self.b), int((y-self.b)/self.b)

    def EXIT(self):
        QCoreApplication.instance().quit(); exit(0)

    def CHEAT(self):
        self.current_player = 0
        self.update()
        self.repaint()

    def SKIP(self):
        self.action = PlayerAction((0,0),(0,0),0); self.state = 2

    def SELECT(self, x, y):
        print(self.state, "Select: (%d,%d)"%(x,y))
        if self.state==0:
            self.selected = (x,y); self.state = 1
        elif self.state==1:
            self.action = PlayerAction(self.selected,(x-self.selected[0],y-self.selected[1]),0); self.state = 2
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
                qp.setPen(QPen(Qt.black, 3+2*(self.state==1 and (i,j)==self.selected)))
                if self.state==1 and (i,j)==self.selected:
                    qp.setBrush(SELECTED)
                elif self.state==1 and PlayerAction(self.selected,(i-self.selected[0],j-self.selected[1]),0).dir_id!=-1:
                    qp.setBrush(COULD_SELECT)
                else:
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

class HumanAgent(BaseAgent):
    def __init__(self, **kwargs):
        self.cheat = kwargs.pop('cheat')
        super(HumanAgent, self).__init__()

    def reset(self, agent_id, **kwargs):
        self.board = kwargs.pop('obs').board
        super(HumanAgent, self).reset(agent_id)
        self.app = QApplication(sys.argv)# ; apply_stylesheet(self.app, theme='light_blue.xml', extra=EXTRA)
        self.gui = GUI(self.board, self.agent_id, self)

    def get_action(self, obs, **info):
        return self.gui.Update(obs.board).serializein(obs.board)