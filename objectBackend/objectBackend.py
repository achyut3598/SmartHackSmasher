from keras.models import model_from_json
from zipfile import ZipFile
from skimage.feature import hog
from sklearn.linear_model import LinearRegression
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from joblib import dump, load

determinationClasses = {1: 'No Car Detected',
                        2: 'Car Appeared',
                        3: 'Car Disappeared',
                        4: 'Staying the same',
                        5: 'Getting Closer',
                        6: 'Getting Further',
                        7: 'Cannot Determine'}

####################################################################
#KITTI Backend Start
def kitti_image_process(myImagePath):
    myImage = cv2.imread(myImagePath)
    resizedImage = cv2.resize(myImage, (256,160))
    resizedImage = [resizedImage]
    return resizedImage

def kitti_local_max(myAxis):
    myMax = 0
    myMaxPos = 0
    myDensity = 0
    myStart = 0
    localMax = []
    zeroCounter = 0
    myTotal = 0
    pos = 0
    for item in myAxis:
        myTotal = myTotal + item[0]
        if item[0] == 0:
            zeroCounter = zeroCounter + 1
        else:
            zeroCounter = 0
            myDensity = myDensity + 1
            if(myMax < item[0]):
                myMax = item[0]
                myMaxPos = pos
            if(myStart == 0):
                myStart = pos

        pos = pos + 1
        if zeroCounter >= 5:
            if(myMax != 0):
                localMax.append([myMax, myMaxPos, myDensity, myStart, myTotal])
                myMax = 0
                myMaxPos = 0
                myDensity = 0
                myStart = 0
                myTotal = 0
    myFinalMaxes = []
    for myLocalMax in localMax:
        if(myLocalMax[0] > 25 and myLocalMax[2] > 25):
            myFinalMaxes.append(myLocalMax)

    return myFinalMaxes

def kitti_pred(myModel, myImage):
    predImage = np.float32(np.stack((myImage), 0)/255.0)
    myPred = myModel.predict(predImage)
    return myPred

def kitti_get_areas(kittiX, kittiY):
    if(len(kittiX) > len(kittiY)):
        myAxis = kittiX
    elif(len(kittiX) < len(kittiY)):
        myAxis = kittiY
    else:
        xVal = 0
        yVal = 0
        for x in kittiX:
            for y in kittiY:
                if(x[4] > y[4]):
                    xVal = xVal + 1
                else:
                    yVal = yVal + 1
        if(xVal > yVal):
            myAxis = kittiY
        else:
            myAxis = kittiX
    if(len(kittiX) == len(kittiY) == 0):
        return 0
    else:
        return myAxis[0][4]
    return 0

def kitti_load_model(myJsonPath, myH5Path):
    jsonFile = open(myJsonPath,'r')
    jsonModel = jsonFile.read()
    jsonFile.close()
    kittiModel = model_from_json(jsonModel)
    kittiModel.load_weights(myH5Path)
    return kittiModel

def kitti_car_status(myAreas):
    carAreas = myAreas
    #print(carAreas)
    if checkForNoCarDetected(carAreas):
        return 1
    elif checkForCarAppeared(carAreas):
        return 2
    elif checkForCarDisappeard(carAreas):
        return 3
    else:
        slope = getSlope(carAreas)
        if slope == 0:
            return 4
        elif slope == -1:
            return 5
        elif slope == 1:
            return 6
        else:
            return 7

def kitti_handler(myModel, myFileDir):
    #print(myFileDir)
    fileNames = glob.glob(myFileDir)
    #print(fileNames)
    kittiAreaList = []
    kittiDeterminations = []
    for imagePath in fileNames:
        kittiPred = kitti_pred(myKittiModel, kitti_image_process(imagePath))
        im = np.array(255 * kittiPred[0], dtype=np.uint8)
        axis0 = np.count_nonzero(im, axis=0)
        axis1 = np.count_nonzero(im, axis=1)
        kittiX = kitti_local_max(axis0)
        kittiY = kitti_local_max(axis1)
        myAreas = kitti_get_areas(kittiX, kittiY)
        kittiAreaList.append(myAreas)
    if(len(kittiAreaList) >= 6):
        kittinum = 0
        while(kittinum <= (len(kittiAreaList)-6)):
            myDet = kitti_car_status(kittiAreaList[kittinum:kittinum+6])
            kittiDeterminations.append(myDet)
            kittinum = kittinum + 1
    # Display Matplotlib of Heatmap
    # plt.figure(figsize=(8, 3))
    # plt.subplot(1, 3, 1)
    # plt.imshow(im)
    # plt.show()
    return kittiDeterminations


#KITTI Backend End
####################################################################

####################################################################
#HOGModel Backend Start

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualize=vis, feature_vector=feature_vec)
        return features

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, orient,
              pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all_rectangles=False):
    # array of rectangles where cars were detected
    rectangles = []

    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]

    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else:
        ctrans_tosearch = np.copy(img)

    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    # select colorspace channel for HOG
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]
    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) + 1  # -1
    nyblocks = (ch1.shape[0] // pix_per_cell) + 1  # -1
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    if hog_channel == 'ALL':
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            test_prediction = svc.predict([hog_features])

            if test_prediction == 1 or show_all_rectangles:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                rectangles.append(
                    ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
    return rectangles

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def process_frame(img,svc):
    rectangles = []
    colorspace = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"

    ystart = 400
    ystop = 464
    scale = 1.0
    rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 416
    ystop = 480
    scale = 1.0
    rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 496
    scale = 1.5
    rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 432
    ystop = 528
    scale = 1.5
    rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 528
    scale = 2.0
    rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 432
    ystop = 560
    scale = 2.0
    rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 400
    ystop = 596
    scale = 3.5
    rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))
    ystart = 464
    ystop = 660
    scale = 3.5
    rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, None,
                                orient, pix_per_cell, cell_per_block, None, None))

    rectangles = [item for sublist in rectangles for item in sublist]

    heatmap_img = np.zeros_like(img[:, :, 0])
    heatmap_img = add_heat(heatmap_img, rectangles)
    heatmap_img = apply_threshold(heatmap_img, 1)
    labels = label(heatmap_img)
    rects = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        rects.append(bbox)

    return rects
def testImages(fileDir,scv):
    fileNames = glob.glob(fileDir)
    counter = -1
    rectangles = []
    hogDeterminations = []
    for fileName in fileNames:
        rectangles.append(process_frame(cv2.resize(mpimg.imread(fileName),dsize = (960,540)),scv))
        counter += 1
        if counter >= 5:
            #print(counter)
            myDet = printCarStatus(rectangles[counter-5:])
            hogDeterminations.append(myDet)
    return hogDeterminations

def printCarStatus(rectangles):
    carAreas = getCarAreas(rectangles)
    if checkForNoCarDetected(carAreas):
        return 1
    elif checkForCarAppeared(carAreas):
        return 2
    elif checkForCarDisappeard(carAreas):
        return 3
    else:
        slope = getSlope(carAreas)
        if slope == 0:
            return 4
        elif slope == -1:
            return 5
        elif slope == 1:
            return 6
        else:
            return 7

def checkForNoCarDetected(areas):
    noCarDetected = True
    for i in range(len(areas)):
        if areas[i] != 0:
            noCarDetected = False
    return noCarDetected

def checkForCarDisappeard(areas):
    carDetected = False
    carDisappeared = False
    for i in range(len(areas)-1):
        if areas[i] != 0:
            carDetected = True
    if carDetected:
        if areas[-1] == 0:
            carDisappeared = True
    return carDisappeared

def checkForCarAppeared(areas):
    carDetected = False
    carAppeared = False
    for i in range(len(areas)-1):
        if areas[i] != 0:
            carDetected = True
    if not carDetected:
        if areas[-1] != 0:
            carAppeared = True
    return carAppeared

def getCarAreas(rectangles):
    carAreas = []
    for i in range(len(rectangles)):
        if len(rectangles[i]) == 0:
            carAreas.append(0)
        else:
            carAreas.append(
                (rectangles[i][0][1][0] - rectangles[i][0][0][0]) * (rectangles[i][0][1][1] - rectangles[i][0][0][1]))
    return carAreas

def isSameIsh(carAreas, areaMargin, numMargin):
    numSameIsh = 0
    for num in range(0,5):
        if(abs(carAreas[num] - carAreas[num+1]) <= areaMargin):
            #print(abs(carAreas[num] - carAreas[num+1]))
            numSameIsh = numSameIsh + 1
    if numSameIsh >= numMargin:
        return True
    else:
        return False

def getSlope(carAreas):
    #print("Areas")
    #print(carAreas)
    model = LinearRegression().fit([[0],[1],[2],[3],[4],[5]], carAreas)
    rSquaredValue = model.score([[0],[1],[2],[3],[4],[5]],carAreas)
    # say staying the same if it is indecisive
    #print(rSquaredValue)
    if rSquaredValue <.10:
        return 2
    else:
        # The Slope is increasing,car getting closer
        if (isSameIsh(carAreas, 50, 4)):
            return 0
        elif (model.coef_[0] > 0):
            return -1
        else:
            return 1

#HOGMOdel Backend End
####################################################################

def extract_from_zip(myZipPath,fileType):
    newFolderName = myZipPath.replace('.zip', '')
    with ZipFile(myZipPath, 'r') as zip:
        zip.extractall(newFolderName)
    newFolderName = newFolderName + '/*' + fileType
    return newFolderName

def object_results(myKittiResults, myHogResults):
    combinedDets = []
    numDets = len(myKittiResults)
    for i in range(numDets):
        if(myKittiResults[i] == myHogResults[i]):
            combinedDets.append(myKittiResults[i])
        else:
            combinedDets.append(7)

    if combinedDets.count(0) > len(combinedDets)/2:
        objectOverallResult = 0
    elif combinedDets.count(1) > len(combinedDets)/2:
        objectOverallResult = 1
    elif combinedDets.count(2) > len(combinedDets)/2:
        objectOverallResult = 2
    elif combinedDets.count(3) > len(combinedDets)/2:
        objectOverallResult = 3
    elif combinedDets.count(4) > len(combinedDets)/2:
        objectOverallResult = 4
    elif combinedDets.count(5) > len(combinedDets)/2:
        objectOverallResult = 5
    elif combinedDets.count(6) > len(combinedDets)/2:
        objectOverallResult = 6
    else:
        objectOverallResult = 7

    objectAnomaliesRaised = []
    if 2 in combinedDets:
        objectAnomaliesRaised.append(1)
    if 3 in combinedDets:
        objectAnomaliesRaised.append(2)

    #(Determination List, Overall Result, Anomalies Raised)
    objectResultList = [combinedDets, objectOverallResult, objectAnomaliesRaised]
    return(objectResultList)



if __name__ == '__main__':

    newFolderName = extract_from_zip('caranomaly.zip','.png')

    kittiModelPath = 'model.json'
    kittiWeightsPath = 'model.h5'
    myKittiModel = kitti_load_model(kittiModelPath, kittiWeightsPath)
    myKittiResults = kitti_handler(myKittiModel, newFolderName)


    hogModelPath = 'zachObjDet.joblib'
    myHogModel = load(hogModelPath)
    myHogResults = testImages(newFolderName, myHogModel)
    myObjectResults = object_results(myKittiResults, myHogResults)

    print(myKittiResults)
    print(myHogResults)
    print(myObjectResults)
    print(determinationClasses[myObjectResults[0][0]])

    print('All Done')
