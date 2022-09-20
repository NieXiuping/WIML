
import random
import sys
import time
import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from controler import Controler

class MIA(QWidget):

    def __init__(self):
        super().__init__()

        self.graph_maker = Controler()
        self.seed_type = 1  # annotation type
        self.all_datasets = []
        self.initUI()

    def initUI(self):
        self.a = QApplication(sys.argv)

        self.window = QMainWindow() 
        # Setup file menu
        self.window.setWindowTitle('WIA')
        mainMenu = self.window.menuBar() 
        fileMenu = mainMenu.addMenu('&File')

        openButton = QAction(QIcon('exit24.png'), 'Open Image', self.window)
        openButton.setShortcut('Ctrl+O') 
        openButton.setStatusTip('Open a file for segmenting.') 
        openButton.triggered.connect(self.on_open) 
        fileMenu.addAction(openButton)

        saveButton = QAction(QIcon('exit24.png'), 'Save Image', self.window)
        saveButton.setShortcut('Ctrl+S')
        saveButton.setStatusTip('Save file to disk.')
        saveButton.triggered.connect(self.on_save)
        fileMenu.addAction(saveButton)

        closeButton = QAction(QIcon('exit24.png'), 'Exit', self.window)
        closeButton.setShortcut('Ctrl+Q')
        closeButton.setStatusTip('Exit application')
        closeButton.triggered.connect(self.on_close)
        fileMenu.addAction(closeButton)
      
        mainWidget = QWidget() 
 
        annotationButton = QPushButton("Load Image")
        annotationButton.setStyleSheet("background-color:white")
        annotationButton.clicked.connect(self.on_open)

        segmentButton = QPushButton("Segment")
        segmentButton.setStyleSheet("background-color:white")
        segmentButton.clicked.connect(self.on_segment)

        NextButton = QPushButton("Save Segmentation")
        NextButton.setStyleSheet("background-color:white")
        NextButton.clicked.connect(self.on_save)

        StateLine = QLabel()
        StateLine.setText("1) Image Input.")
        palette = QPalette() 
        palette.setColor(StateLine.foregroundRole(), Qt.blue)
        StateLine.setPalette(palette)

        MethodLine = QLabel()
        MethodLine.setText("2) Initial Segmentation.")
        mpalette = QPalette()
        mpalette.setColor(MethodLine.foregroundRole(), Qt.blue)
        MethodLine.setPalette(mpalette)

        NoteLine = QLabel()
        NoteLine.setText("Note: blue/green contour: GT/Pred. ")
        mpalette = QPalette()
        mpalette.setColor(NoteLine.foregroundRole(), Qt.black)
        NoteLine.setPalette(mpalette)

        RefineLine = QLabel()
        RefineLine.setText("3) Refine: clinician click. ")
        mpalette = QPalette()
        mpalette.setColor(RefineLine.foregroundRole(), Qt.blue)
        RefineLine.setPalette(mpalette)

        Note2Line = QLabel()
        Note2Line.setText("Note: left/right mouse click:\nunder-segmentation/over-segmentation. ")
        mpalette = QPalette()
        mpalette.setColor(Note2Line.foregroundRole(), Qt.black)
        Note2Line.setPalette(mpalette)

        SaveLine = QLabel()
        SaveLine.setText("4) Save.")
        spalette = QPalette()
        spalette.setColor(SaveLine.foregroundRole(), Qt.blue)
        SaveLine.setPalette(spalette)

        hbox = QVBoxLayout()
        hbox.addWidget(StateLine)
        hbox.addWidget(annotationButton)
        hbox.addWidget(MethodLine)
        hbox.addWidget(segmentButton)
        hbox.addWidget(NoteLine)
        hbox.addWidget(RefineLine)
        hbox.addWidget(Note2Line)
        hbox.addWidget(SaveLine)
        hbox.addWidget(NextButton)
        hbox.addStretch()  

        tipsFont = StateLine.font()
        tipsFont.setPointSize(10)
        StateLine.setFixedHeight(15)
        StateLine.setWordWrap(True) 
        StateLine.setFont(tipsFont)
        MethodLine.setFixedHeight(15)
        MethodLine.setWordWrap(True)
        MethodLine.setFont(tipsFont)
        SaveLine.setFixedHeight(15)
        SaveLine.setWordWrap(True)
        SaveLine.setFont(tipsFont)


        self.seedLabel = QLabel()
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))
        self.seedLabel.mousePressEvent = self.mouse_down
        self.seedLabel.mouseMoveEvent = self.mouse_drag
    
        imagebox = QHBoxLayout() 
        imagebox.addWidget(self.seedLabel)

        vbox = QHBoxLayout()
        vbox.addLayout(imagebox)
        vbox.addLayout(hbox)

        mainWidget.setLayout(vbox)

        self.window.setCentralWidget(mainWidget)
        self.window.show()

    @staticmethod
    def get_qimage(cvimage):
        height, width, bytes_per_pix = cvimage.shape
        bytes_per_line = width * bytes_per_pix
        cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB, cvimage)
        return QImage(cvimage.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def mouse_down(self, event):
        if event.button() == Qt.LeftButton:
            self.seed_type = 2 #foreground
        elif event.button() == Qt.RightButton:
            self.seed_type = 3 #background
        self.graph_maker.add_seed(event.x(), event.y(), self.seed_type)
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))
        
        #refinement
        self.graph_maker.refined_seg()
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.refined_seg))))
        
    def mouse_drag(self, event):
        self.graph_maker.add_seed(event.x(), event.y(), self.seed_type)
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))

    @pyqtSlot()
    def on_open(self):
        f = QFileDialog.getOpenFileName() 
        if f[0] is not None and f[0] != "":
            f = f[0]
            self.graph_maker.load_image(str(f))
            self.seedLabel.setPixmap(QPixmap.fromImage(
                self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.seeds))))
        else:
            pass

    def on_save(self):
        self.graph_maker.save_image()
        print('Save successful!')
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.save_image))))
       
    @pyqtSlot()
    def on_close(self):
        print('Closing')
        self.window.close()

    @pyqtSlot()
    def on_segment(self):
        self.graph_maker.cam_segmentation()
        self.seedLabel.setPixmap(QPixmap.fromImage(
            self.get_qimage(self.graph_maker.get_image_with_overlay(self.graph_maker.cam_segmentation))))

 
