from django.shortcuts import render, redirect   
from django.http import HttpResponse, StreamingHttpResponse
from emo import views
from django.views.generic import View
from django.views.decorators import gzip

from model import FacialExpressionModel
import cv2
import numpy as np
import os

from os import listdir
from os.path import isfile, join
import shutil

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import shutil 

import json
import pdb
import base64

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]


model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

cwd = os.getcwd()


# from emotion import Emotion

class Base(View):

    def get(self, request, *args, **kwargs):

        return render(request,'base.html', context={})
    def post(self, request, *args, **kwargs):

        return render(request,'base.html', context={})
class Emotion(View):

    def get(self, request, *args, **kwargs):

        return render(request,'base.html', context={})
    def post(self, request, *args, **kwargs):
        rearrange = emotion = train = False
        # print(request.POST['rearrange'])
        # emo = request.POST['rearrange']
        if "emotion" in request.POST:
            emotion = True
        if "train" in request.POST:
            train = True
        if emotion:
            print ("emotion")
            context = {
                'name' : 'emotion'
            }
        if train:
            print ("train")
            context = {
                'name' : 'train'
            }

        return render(request,'emotion.html', context=context)

class Detect_emotion:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.face_count = 0

    def __del__(self):
        self.video.release()


    def emo(self): 
        # pdb.set_trace()
        
        ret, img = self.video.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3,5)

        for(x,y,w,h) in face:
            self.face_count += 1
        
            face_color = img[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]

            # resize image
            roi = cv2.resize(face_gray, (48, 48))

            # predict image
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            #save emotions to the folder
            cv2.imwrite("static/Emotion/%s/%s_%d.jpg"%(pred,pred,self.face_count), face_color)

            # draw emotion into frame
            cv2.putText(img, pred, (x, y), font, 1, (255, 255, 0), 2)

            # draw rectangle around face
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255,0), 2)

            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()
            # cv2.destroyAllWindows()



def gen(camera):
    while True:
        frame = camera.emo()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page
def video(request):
    try:
        return StreamingHttpResponse(gen(Detect_emotion()),
                content_type='multipart/x-mixed-replace; boundary=frame')
    except:
        print("Aborted")

from PIL import Image
import base64
from io import BytesIO

from datetime import datetime


class Restapi_predict(View):
    def __init__(self):
        self.face_count = 0
    def get(self, request, *args, **kwargs):
        return render(request, 'rearrange.html', context={})

    def post(self, request, *args, **kwargs):
        # pdb.set_trace()
        img_string = request.POST['frame']
        # frame = cv2.imread('sad.jpg')
        
        face_st = Image.open(BytesIO(base64.b64decode(img_string)))
        frame = np.asarray(face_st)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.3,5)
        print ("face", face)
        if len(face) > 0:
            for(x,y,w,h) in face:
                self.face_count += 1
                face_gray = gray[y:y+h, x:x+w]
                face_color = frame[y:y+h, x:x+w]

                # resize image
                roi = cv2.resize(face_gray, (48, 48))

                # predict image
                pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                now = datetime.now()
                time = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
                #save emotions to the folder
                cv2.imwrite("static/Emotion/%s/%s.jpg"%(pred,time), face_color)
                context = {'emotion' : pred}
                json_data = json.dumps(context)
        else:
            context = {'error' : "Face not found"}
            json_data = json.dumps(context)
        return HttpResponse(json_data, content_type='application/json')



class Rearrange(View):

    def post(self, request, *args, **kwargs):
        # pdb.set_trace()
        emo = None
        files = []
        if 'emo' in request.POST:
            emo = request.POST['emo']
            print(emo)
            mypath = os.path.join(cwd,'static','Emotion',emo)

            files = os.listdir(mypath)
        context = {"emo": emo,
            'images':files,
             }
        return render(request,'rearrange.html', context=context)

    def get(self, request, *args, **kwargs):
            
        return render(request, 'rearrange.html',context={})




# import pdb
class Move_images(View):   
    def get(self, request, *args, **kwargs):
        return render(request, 'rearrange.html', context={})

    # def post(self, request, *args, **kwargs):
    #     # pdb.set_trace()
    #     move = request.POST
    #     print(move)
    #     return render(request, 'rearrange.html', context={})
    def post(self, request, *args, **kwargs):
        move = request.POST
        print(move)

        img_fr = request.POST['from']
        img_to = request.POST['emo']
        imgs = request.POST.getlist('image')

        for i in imgs:
            source = os.path.join(cwd,'static','Emotion', img_fr, i)
            destination = os.path.join(cwd,'static','Emotion', img_to, i)
            shutil.move(source, destination)

        return redirect('rearrange')

import json
# @csrf_exempt
class Restapi_move(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'rearrange.html', context={})

    def post(self, request, *args, **kwargs):
        data = request.POST
        emo_fr = request.POST['image_from']
        emo_to = request.POST['image_to']
        img_list = request.POST.getlist('images')

        for i in img_list:
            source = os.path.join(cwd,'static','Emotion', emo_fr, i)
            destination = os.path.join(cwd,'static','Emotion', emo_to, i)
            shutil.move(source, destination) 
        context = {'msg':'Images successfully moved'}
        json_data = json.dumps(context)
        return HttpResponse(json_data, content_type='application/json')



from .train import train_data_obj
from keras.models import model_from_json

class Train_model(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'base.html', context={})

    def post(self, request, *args, **kwargs):
        if "train" in request.POST:
            train = request.POST['train']
            print(train)
            for i in EMOTIONS_LIST:
                source = os.path.join(cwd,'static','Emotion',i)
                destination_tr = os.path.join(cwd,'static','train',i)
                destination_val = os.path.join(cwd,'static','validation',i)
                files = os.listdir(source)
                t_len = len(files)
                train_img = files[:int((t_len*80)/100)]
                valid_img = files[int((t_len*80)/100):]
                for x in train_img:
                    shutil.copy(os.path.join(source,x),destination_tr)
                    
                for y in valid_img:
                    shutil.copy(os.path.join(source,y),destination_val)    
                        
            train_gen, val_gen = train_data_obj.data_generation()
            train_data_obj.transfer_learn( train_gen, val_gen)


        return redirect('base')

def get(self, request, *args, **kwargs):
        return render(request, 'rearrange.html', context={})































