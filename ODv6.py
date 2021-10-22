import cv2 as cv
import numpy as np
import cv2
import imutils
from imutils.video import FPS
import time



confThreshold = 0.25
#confThreshold = 0.4
nmsThreshold = 0.40
#nmsThreshold = 0.50
inpWidth = 416
inpHeight = 416


classesFile = "coco.names"
classes = None

with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


modelConf = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []
    


    

    for out in outs:
        for detection in out:
            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            

            if confidence > confThreshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes (boxes,confidences, confThreshold, nmsThreshold )

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)


def drawPred(classId, conf, left, top, right, bottom):
    
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    label = '%.2f' % conf

   
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    
    cv.putText(frame, label, (left,top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def getOutputsNames(net):
    
    layersNames = net.getLayerNames()
   
    
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# this is where the net is setup I am curretn trying to compile open-cv to have the target be CUDA
# instead of CPU, but im not sure if this will be enough to do the trick, the nvdia nano also has a make file which has certain presets such as GPU=1

net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
#net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)




winName = 'DL OD with OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 1000,700)





cap = cv2.VideoCapture('jenoptik.mp4')

fps = FPS().start()

start_time = time.time()
x = 1   # displays the frame rate every 1 second
counter = 0



while cv.waitKey(1) < 0:
 #while True:
   
  
   
    
    hasFrame, frame = cap.read()
 
	
    counter+=1
    if (time.time() - start_time) > x :
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
    
    
	
    
    
    
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    fps.update()
    
    
    
    
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)

   
    net.setInput(blob)
    outs = net.forward (getOutputsNames(net))


    postprocess (frame, outs)

   
    cv.imshow(winName, frame)
fps.stop()



















