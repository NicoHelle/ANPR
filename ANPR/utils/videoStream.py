
from ultralytics import YOLO
#import test as arch
import os.path as osp
import utils.RRDBNet_arch as arch
#from RRDBNet_arch import RRDBNet_arch as arch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch

import easyocr
from PIL import Image
import glob
#from utils.test import hallo
import torch
import utils.RRDBNet_arch as arch
import numpy as np
#hallo() 
#bla.testen()

i = 0

reader = easyocr.Reader(['en'], gpu=True)
model = YOLO("C:/Users/herol/Desktop/Kennzeichen-20230721T102328Z-001/Kennzeichen/utils/best.pt")
model.to('cuda')
model_path = 'C:/Users/herol/Desktop/Kennzeichen-20230721T102328Z-001/Kennzeichen/utils/RRDB_PSNR_x4.pth'
#model_path = 'C:/Users/User/Desktop/Kennzeichen/utils/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

srmodel = arch.RRDBNet(3, 3, 64, 23, gc=32)
srmodel.load_state_dict(torch.load(model_path), strict=True)
srmodel.eval()
srmodel = srmodel.to(device)
conf = 0.5
kenzeichenList = []

sr = cv2.dnn_superres.DnnSuperResImpl_create()
pathESPCN = "C:/Users/herol/Desktop/Kennzeichen-20230721T102328Z-001/Kennzeichen/utils/ESPCN_x4.pb"
sr.readModel(pathESPCN)
sr.setModel("espcn", 4) # set the model by passing the value and the upsampling ratio

def superResCV2(image):

    
    result = sr.upsample(image) # upscale the input image
    
    return result



def image_preprocessing(cropped_image):
    
    #plt.imshow(cropped_image)
    #plt.show()
    
    #print("grop_characters")
    
    gray = cropped_image
    
    #gray = cv2.resize(cropped_image, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    
    #plt.imshow(gray)
    #plt.show()
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)             
    
    #plt.imshow(blur)
    #plt.show()
    
    #bilateral_blur = cv2.bilateralFilter(blur, 15, 80, 80)
    
    
    #gray = bilateral_blur
    
    #plt.imshow(gray)
    #plt.show()
    
    #gray = cv2.medianBlur(bilateral_blur, 3)
    
    #plt.imshow(gray)
    #plt.show()
    
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    #plt.imshow(thresh)
    #plt.show()
    return thresh

def superres(img):
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = srmodel(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    return output

def setConf(tresh):

    conf = tresh
    print(tresh)
    print(str(conf)+"hhsfdf")

def imageTest(image, tresh):    
    #if i >= 0:
    global i, kenzeichenList
    #kenzeichenList =[]
    prediction = model(image,conf=tresh, verbose=False)
    res_plotted = prediction[0].plot() 
    #print(prediction) 
    for r in prediction:
        #print("for 1")
        boxes = r.boxes
        for box in boxes:
            #print("for 2")  
            b = box.xyxy[0] 
            c = box.cls
            cords = b.cpu().data.numpy()
            #print(cords)
            cords = (np.ceil(cords)).astype(int)
            start_point = (cords[0], cords[1])
            end_point = (cords[2], cords[3])
            color = (0, 255, 0)
            thickness = 2
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            (x1, y1) = (cords[0], cords[1])
            (x2, y2) = (cords[2], cords[3])    
            cropped_image = image[y1:y2, x1:x2]
            #cv2.imwrite('./utils/output/image' + str(i) + '.png', image)
            #cv2.imwrite('./utils/output/cropped' + str(i) + '.png', cropped_image)


            #scropped_image = superResCV2(cropped_image)
            #cropped_image = superres(cropped_image) 
            #cv2.imwrite('./utils/output/croppedSuper' + str(i) + '.png', cropped_image)  
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            ocr_result =[]
            #try:
            ocr_result = reader.readtext(gray)
            #except:
                #print("fehler")


            kennzeichen1 = ""
                
            for result in ocr_result:
                print("ocr_result 1: " + result[1])
                kennzeichen1 += result[1]
                    
            gray = image_preprocessing(gray)
                    
            ocr_result = reader.readtext(gray)

            font = cv2.FONT_HERSHEY_SIMPLEX

            org = (x1, y1-20)

            fontScale = 1

            color = (255, 0, 0)

            thickness = 2
                    
            kennzeichen2 = ""
                    
            for result in ocr_result:
                print("ocr_result 2: " + result[1])
                kennzeichen2 += result[1]
                    
            image = cv2.putText(image, "Test1: " + str(kennzeichen1) + " Test2: " + str(kennzeichen2), org, font, fontScale, color, thickness, cv2.LINE_AA) 
    
            #kenzeichenList.append(kennzeichen1)
            #cv2.imwrite('./utils/output/ergebnis' + str(i) + '.png', image)
            i +=1
    return image #, kenzeichenList

            

    

