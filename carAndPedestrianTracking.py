import cv2
from random import *



#getting webcam (u can add any vieo instead of 0)
video = cv2.VideoCapture('video2.avi')
#load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_car_data = cv2.CascadeClassifier("cars.xml")
#load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_fullbody_data = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_fullbody.xml")

#iterate forever over frames
while True:
    #Read the current frame
    (successful_frame_read, frame) = video.read()
    if successful_frame_read:
        #grayscaled_image
        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    #detect faces
    face_coordinates = trained_car_data.detectMultiScale(grey_image)

    #detect full body
    body_coordinates = trained_fullbody_data.detectMultiScale(grey_image)

    #draw a rectangle around the car
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)
    #draw a rectangle around the fullbody
    for (x,y,w,h) in body_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

    #display the image with rectangle around the face
    cv2.imshow('Car and pedestrian tracking',frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break
video.release()


"""
car_image = cv2.imread('car.jpeg')

#our pretratined data
classifier_file = 'cars.xml'
#load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_car_data = cv2.CascadeClassifier("cars.xml")

#grayscaled_image
grey_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)

#detect faces
car_coordinates = trained_car_data.detectMultiScale(grey_image)

#draw a rectangle around the image
for (x,y,w,h) in car_coordinates:
    cv2.rectangle(car_image,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

#display the image with rectangle around the face
cv2.imshow('car',car_image)
cv2.waitKey()
print(car_coordinates)
"""




print("Code Completed")