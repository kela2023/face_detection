import cv2
import numpy as np
from keras.models import load_model

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video=cv2.VideoCapture(0)
video.set(3, 640)
video.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX


model = load_model('keras_model.h5')
model_genre = load_model('./model-013.model')
classifier = load_model('model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
#genre
labels_dict={0:'Male',1:'Female'}
color_dict={0:(0,0,255),1:(0,255,0)}

#classification de visages
def get_className(classNo):
	if classNo==0:
		return "Silue"
	elif classNo==1:
		return "Kela"


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class Videeo(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,frame=self.video.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1,3,5)
        for x,y,w,h in faces:
            x1,y1=x+w, y+h
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),1)
            cv2.line(frame,(x,y),(x+30,y),(255,0,255),6)
            cv2.line(frame,(x,y),(x,+30),(255,0,255,6))

            cv2.line(frame,(x1,y),(x1-30,y),(255,0,255),6)
            cv2.line(frame,(x1,y),(x1,y+30),(255,0,255),6)

            cv2.line(frame,(x,y),(x+30,y1),(255,0,255),6)
            cv2.line(frame,(x,y1),(x,y1-30),(255,0,255),6)

            cv2.line(frame,(x1,y1),(x1-30,y1),(255,0,255),6)
            cv2.line(frame,(x1,y1),(x1,y1-30),(255,0,255),6)
            
            crop_img=imgOrignal[y:y+h,x:x+h]
            img=cv2.resize(crop_img, (224,224))
            img=img.reshape(1, 224, 224, 3)
            prediction=model.predict(img)
            classIndex=np.argmax(prediction)
            probabilityValue=np.amax(prediction)

            if classIndex==0:
                cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
                cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
            elif classIndex==1:
                cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,0,255),-2)
                cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)

        #   	cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA)
                face_img=gray[y:y+w,x:x+w]#begin
                resized=cv2.resize(face_img,(32,32))
                normalized=resized/255.0
                reshaped=np.reshape(normalized,(1,32,32,1))
                result=model_genre.predict(reshaped)
                label=np.argmax(result,axis=1)[0]
                label_fr=np.argmax(result,axis=1)[0]
                cv2.putText(imgOrignal, labels_dict[label], (x, y-40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        #end
        #ella
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)
                    prediction = classifier.predict(roi)[0]
                    label=emotion_labels[prediction.argmax()]
                    label_position = (x,y)
                    cv2.putText(imgOrignal,label,(x,y-80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                else:
                    cv2.putText(imgOrignal,'No Faces',(x,y-110),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()