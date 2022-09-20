
import os
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from gui import MIA


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MIA()
    sys.exit(app.exec_())
