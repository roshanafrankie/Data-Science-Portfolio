import sys
import numpy as np
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

class Ui_MainWindow(object):
    def __init__(self):
        # 1. Setup Pathing
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'my_model.h5')

        # 2. Reconstruct Model Architecture 
        # (This avoids the "batch_shape" metadata error entirely)
        if os.path.exists(model_path):
            print(f"Success! Found model at: {model_path}")
            try:
                # We manually build the "skeleton" of the Road Sign model
                self.model = Sequential()
                self.model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(30, 30, 3)))
                self.model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
                self.model.add(MaxPool2D(pool_size=(2, 2)))
                self.model.add(Dropout(rate=0.25))
                self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
                self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
                self.model.add(MaxPool2D(pool_size=(2, 2)))
                self.model.add(Dropout(rate=0.25))
                self.model.add(Flatten())
                self.model.add(Dense(256, activation='relu'))
                self.model.add(Dropout(rate=0.5))
                self.model.add(Dense(43, activation='softmax'))

                # Now we load ONLY the weights into this skeleton
                # This skips the problematic 'load_model' metadata check
                self.model.load_weights(model_path)
                print("Architecture reconstructed and weights loaded successfully!")
            except Exception as e:
                print(f"Error loading weights: {e}")
                print("Tip: Make sure your training notebook used the same 30x30 architecture.")
                self.model = None
        else:
            print(f"CRITICAL ERROR: Model file 'my_model.h5' not found.")
            self.model = None

        # 3. Class Names Mapping
        self.class_names = { 
            0:"Speed limit (20km/h)", 1:"Speed limit (30km/h)", 2:"Speed limit (50km/h)",
            3:"Speed limit (60km/h)", 4:"Speed limit (70km/h)", 5:"Speed limit (80km/h)",
            6:"End of speed limit (80km/h)", 7:"Speed limit (100km/h)", 8:"Speed limit (120km/h)",
            9:"No passing", 10:"No passing veh over 3.5 tons", 11:"Right-of-way at intersection",
            12:"Priority road", 13:"Yield", 14:"Stop", 15:"No vehicles",
            16:"Veh > 3.5 tons prohibited", 17:"No entry", 18:"General caution",
            19:"Dangerous curve left", 20:"Dangerous curve right", 21:"Double curve",
            22:"Bumpy road", 23:"Slippery road", 24:"Road narrows on the right",
            25:"Road work", 26:"Traffic signals", 27:"Pedestrians", 28:"Children crossing",
            29:"Bicycles crossing", 30:"Beware of ice/snow", 31:"Wild animals crossing",
            32:"End speed + passing limits", 33:"Turn right ahead", 34:"Turn left ahead",
            35:"Ahead only", 36:"Go straight or right", 37:"Go straight or left",
            38:"Keep right", 39:"Keep left", 40:"Roundabout mandatory",
            41:"End of no passing", 42:"End no passing veh > 3.5 tons" 
        }

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        
        self.BrowseImage = QtWidgets.QPushButton("Browse Image", self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)
        
        self.Classify = QtWidgets.QPushButton("Classify", self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 390, 211, 51))
        self.textEdit.setReadOnly(True)
        
        self.label_title = QtWidgets.QLabel("ROAD SIGN RECOGNITION", self.centralwidget)
        self.label_title.setGeometry(QtCore.QRect(110, 20, 621, 20))
        font = QtGui.QFont("Courier New", 14, QtGui.QFont.Bold)
        self.label_title.setFont(font)

        MainWindow.setCentralWidget(self.centralwidget)
        
        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)

    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *jpeg *.bmp)")
        if fileName:
            self.file = fileName
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio)
            self.imageLbl.setPixmap(pixmap)
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)

    def classifyFunction(self):
        if self.model is None:
            self.textEdit.setText("Model Error")
            return
        if not hasattr(self, 'file'):
            self.textEdit.setText("Select Image First")
            return
            
        test_image = Image.open(self.file)
        test_image = test_image.resize((30, 30))
        test_image = np.array(test_image)
        # Ensure it is float32 and normalized if your training did that
        test_image = test_image.astype('float32') / 255
        test_image = np.expand_dims(test_image, axis=0)

        result = self.model.predict(test_image)[0]
        sign = self.class_names[result.argmax()]
        self.textEdit.setText(sign)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())