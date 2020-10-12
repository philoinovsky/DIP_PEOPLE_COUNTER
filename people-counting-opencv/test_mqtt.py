import paho.mqtt.client as mqtt
from time import sleep
import cv2
import imutils
import base64
import sys
client = mqtt.Client()
client.connect("128.199.208.181", 1883, 60)
print("connecting to server...")
sleep(2)
video = "./videos_and_pics/example_01.mp4"
vs = cv2.VideoCapture(video)
newwidth = 500
# newheight = 400
FPS = 8
skipframe = 0
for i in range(10000000000000000):
    sys.stdout.flush()
    sys.stdout.write("\rframe %d" % i)
    success,frame = vs.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=newwidth)
    retval, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer)
    client.publish("camera",jpg_as_text)
    sleep(1/FPS)
    for _ in range(skipframe):
        vs.read()
print("\nfinished")