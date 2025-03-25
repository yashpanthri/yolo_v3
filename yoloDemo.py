import cv2
import numpy as np 

cap = cv2.VideoCapture(0) #If you want to use webcam, you can use 0 or 1 or 2. 0 is for default webcam.
whT = 320
classesFilePath = 'coco.names'
classNames =[]
with open(classesFilePath, 'rt') as f:
    classNames = f.read().strip('\n').split('\n')

# print (classNames)

modelConfiguration = 'yolov3.cfg' #yolov3.cfg, yolov3-tiny.cfg, yolov3-spp.cfg
# modelConfiguration = 'yolov3-tiny.cfg' 
modelWeights = 'yolov3.weights' #yolov3.weights, yolov3-tiny.weights, yolov3-spp.weights
# modelWeights = 'yolov3-tiny.weights'
confTreashold = 0.5
nmsThreshold = 0.3
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights) #Reads the network model stored in Darknet model files.
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) #If you want to use GPU, you can use cv2.dnn.DNN_BACKEND_CUDA
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) #If you want to use GPU, you can use cv2.dnn.DNN_TARGET_OPENCL


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = [] #Bounding box
    classIds = [] #Class Ids
    confs = [] #Confidence
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confTreashold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confTreashold, nmsThreshold) #Performs non maximum suppression given boxes and corresponding scores.
    print(indices)
    for i in indices:  
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT,whT), [0,0,0], 1, crop = False) #Creates 4-dimensional blob from image with (batch size, channels, height, width)
    net.setInput(blob) #Sets the new input value for the network.
    layerNames = net.getLayerNames() #Returns names of layers of the network.
    # print(layerNames)
    # print(net.getUnconnectedOutLayers()) #Returns indexes of layers with unconnected outputs.
    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()] #Returns names of output layers of the network.
    # print(outputNames) #['yolo_82', 'yolo_94', 'yolo_106']
    outputs = net.forward(outputNames) #Runs forward pass to compute output of layer with name outputName.
    # print(len(outputs))
    # print(type(outputs[0])) #<class 'numpy.ndarray'>
    # print(outputs[0].shape) #(300, 85)
    # print(outputs[1].shape) #(1200, 85)
    # print(outputs[2].shape) #(4800, 85)
    # print(outputs[0][0]) #

    findObjects(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

 