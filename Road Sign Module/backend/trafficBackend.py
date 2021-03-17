import logging
import numpy
from keras.models import load_model
import pickle
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import numpy as np
import cv2
import imutils
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

#Traffic sign labels
classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}

classesSet = {1: 'Speed',
           2: 'Speed',
           3: 'Speed',
           4: 'Speed',
           5: 'Speed',
           6: 'Speed',
           7: 'End',
           8: 'Speed',
           9: 'Speed',
           10: 'No passing',
           11: 'No passing',
           12: 'Tri',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'Vehicles',
           17: 'Vehicles',
           18: 'Vehicles',
           19: 'Tri',
           20: 'Tri',
           21: 'Tri',
           22: 'Tri',
           23: 'Tri',
           24: 'Tri',
           25: 'Tri',
           26: 'Tri',
           27: 'Tri',
           28: 'Tri',
           29: 'Tri',
           30: 'Tri',
           31: 'Tri',
           32: 'Tri',
           33: 'End',
           34: 'Arrow',
           35: 'Arrow',
           36: 'Arrow',
           37: 'Arrow',
           38: 'Arrow',
           39: 'Arrow',
           40: 'Arrow',
           41: 'Arrow',
           42: 'End',
           43: 'End'}

def semi_model_load(modelPath):
    model = load_model(modelPath)
    return model

def sup_model_load(modelPath):
    model = load_model(modelPath)
    return model

def unsup_model_load(modelPath):
    model = pickle.load(open(modelPath, 'rb'))
    return model

def semi_image_handler(imagePath):
    rawImage = Image.open(imagePath)
    editedImage = rawImage.resize((64, 64))
    data = np.array(editedImage)
    data = np.expand_dims(data, axis=0)
    return data

def sup_image_handler(imagePath):
    size = (256, 256)
    pixels = load_img(imagePath, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    pixels = pixels.reshape(256, 256, 3)
    pixels = (pixels - 127.5)/127.5
    return pixels

def unsup_image_handler(imagePath, bins=(8, 8, 8)):
    # Convert image to histogram
    image = cv2.imread(imagePath)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return [hist.flatten()]

def semi_predict(semiModel, imagePath):
    myData = semi_image_handler(imagePath)
    pred = semiModel.predict_classes(myData)[0]
    return pred

def sup_predict(supModel, imagePath):
    pixels = sup_image_handler(imagePath)
    pred = supModel.predict_classes(pixels)[0]
    return pred

def unsup_predict(unsupModel, imagePath):
    features = unsup_image_handler(imagePath)
    pred = unsupModel.predict(features)[0]
    return int(pred)

def predSign(myPred):
    return classes[myPred + 1]

def predClass(myPred):
    return classesSet[myPred + 1]

def analyze_road_sign(unsupModel, semiModel, supModel, imagePath):
    if unsupModel != None:
        #logging.info("Unsupervised model exists. Prediction Started")
        unsupPred = unsup_predict(unsupModel, imagePath)
    if semiModel != None:
        #logging.info("Semisupervised model exists. Prediction Started")
        semiPred = semi_predict(semiModel, imagePath)
    if supModel != None:
        #logging.info("Supervised model exists. Prediction Started")
        #supPred = sup_predict(supModel, imagePath)
        supPred = 0
    supPred = 0
    if unsupPred == semiPred or unsupPred == supPred or semiPred == supPred:
        combinedVoteSign = True
    else:
        combinedVoteSign = False
    if classesSet[unsupPred+1] == classesSet[semiPred+1] or classesSet[unsupPred+1] == classesSet[supPred+1] or classesSet[semiPred+1] == classesSet[supPred+1]:
        combinedVoteSet = True
    else:
        combinedVoteSet = False


    analysisResults = [combinedVoteSign, combinedVoteSet, unsupPred, semiPred, supPred, imagePath]

    return analysisResults

#Sample Usage
def main():
    #Sets up logging to be "time - name - level - message"
    #logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #logging.info("Start of Program")
    unsupPath = "D:/School/2021/SD/knnHist.sav"
    semiPath = "D:/School/2021/SD/my_model.h5"
    supPath = "D:/School/2021/SD/saved_model.pb"
    imagePath = "D:/School/SD/archive/Test/00261.png"
    #logging.info("Loading Unsupervised Model")
    unsupModel = unsup_model_load(unsupPath)
    #logging.info("Loading Semisupervised Model")
    semiModel = semi_model_load(semiPath)
    #logging.info("Loading Supervised Model")
    #supModel = sup_model_load(supPath)
    supModel = None
    results = analyze_road_sign(unsupModel, semiModel, supModel, imagePath)
    print(results)

if __name__ == "__main__":
    main()
