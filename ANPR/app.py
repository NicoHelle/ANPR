from flask import Flask, render_template, Response, request,redirect, url_for, jsonify
import cv2

from flask import jsonify
import time
import cv2
import torch
from PIL import Image
import sys
import matplotlib.pyplot as plt
from utils.videoStream import imageTest, setConf
#from utils.videoStream import * 
 
app = Flask(__name__)

#Model laden


#video
videofeed = cv2.VideoCapture("C:/Users/herol/Desktop/Kennzeichen-20230721T102328Z-001/Kennzeichen/utils/w.mp4")
#webcam
#videofeed = cv2.VideoCapture(0)

kenn =[]
#mode = 0
threshold = 0.5
#use_gpu = False
#current_model = 'SGSC'
seconds = 0.0

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""
    global seconds
    z=0
    while(videofeed.isOpened()):
      # Capture frame-by-frame
        ret, img = videofeed.read()
        z+=1
        if z==1:
            z = 0
            if ret == True:
                start_prediction = time.time()
                #frame= imageTest(img)
                #frame,kenn= imageTest(img,threshold)
                frame= imageTest(img,threshold)
                end_prediction = time.time()
                #print(threshold)
                seconds = end_prediction - start_prediction 
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                #yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.1)
            else:
                break
           # z+=1
        
    


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    #v1, v2 = gen()
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')










@app.route("/threshold", methods=['GET', 'POST'])
def set_threshold():
    global threshold
    
    if request.method == "POST":
        threshold = float(request.json['data'])
        



    return render_template('index.html') #redirect(url_for('index'))







@app.route("/performance", methods=['GET'])
def get_performance():
    global seconds
    return jsonify({'speed': seconds})








if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
