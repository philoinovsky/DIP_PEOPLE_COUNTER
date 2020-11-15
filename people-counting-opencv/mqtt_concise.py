import numpy as np
import dlib, cv2, imutils
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from pathlib import Path
import paho.mqtt.client as mqtt
import argparse, sys, time, base64, math, requests, json, threading

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="#path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="#path to Caffe pre-trained model")
ap.add_argument("-o", "--output", type=str,
    help="#path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
    help="#minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
    help="# of skip frames between detections")
args = vars(ap.parse_args())
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
writer = None
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
totalFrames = 0
totalDown = 0
totalUp = 0
fps = FPS().start()
counter = 0
delta = 0

def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t

def updateoccupancy():
    level, room = ("s2_b4","s2_b4c_17")
    global delta
    print("number of people + " + str(delta))
    data = requests.get("http://127.0.0.1:8000/"+level).json()
    id = -1
    for i, v in enumerate(data):
        if v["name"] == room:
            id = i
            data[i]["occupied"] = data[i]["occupied"] + delta
            break
    link = "http://127.0.0.1:8000/" + level + "/" + room
    delta = 0
    data[i]['location'] = level
    headers = {'Authorization': 'Token e770960caa67450b109b2df12fc043012bc1abe6'}
    # print(data[i])
    return requests.put(url=link, data=json.dumps(data[i]), headers=headers)


# def decodeb64(jpg_as_text):
#     jpg_original = base64.b64decode(jpg_as_text)
#     jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
#     image_buffer = cv2.imdecode(jpg_as_np, flags=1)

#     ht, wd, cc= image_buffer.shape
#     # create new image of desired size and color (blue) for padding
#     ww = 1000
#     hh = 750
#     color = (0,0,0)
#     result = np.full((hh,ww,cc), color, dtype=np.uint8)

#     # compute center offset
#     xx = (ww - wd) // 2
#     yy = (hh - ht) // 2

#     # copy image_buffer image into center of result image
#     result[yy:yy+ht, xx:xx+wd] = image_buffer
#     # return image_buffer
#     return result

def decodeb64(jpg_as_text):
    jpg_original = base64.b64decode(jpg_as_text)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    image_buffer = cv2.imdecode(jpg_as_np, flags=1)
    return image_buffer

def on_message(client, userdata, msg):
    # read image as base64 from MQTT
    # print(msg.topic+" "+str(msg.payload))
    global net, writer, ct, trackers, trackableObjects, totalFrames, \
        totalDown, totalUp, fps, counter, delta
    # sys.stdout.flush()
    # sys.stdout.write("[INFO] receive frame %d\n" % counter)
    sys.stdout.write("#")
    counter += 1
    jpg_as_text = msg.payload
    frame = decodeb64(jpg_as_text)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (H, W) = frame.shape[:2]
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (W, H), True)
    status = "Waiting"
    rects = []
    if totalFrames % args["skip_frames"] == 0:
        status = "Detecting"
        trackers = []
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                trackers.append(tracker)
    else:
        for tracker in trackers:
            status = "Tracking"
            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            rects.append((startX, startY, endX, endY))
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
    objects = ct.update(rects)
    coordinatex = []
    coordinatey = []
    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            if not to.counted:
                if direction < 0 and centroid[1] < H // 2:
                    # up -> user leaving, minus delta by 1
                    delta -= 1
                    totalUp += 1
                    to.counted = True
                elif direction > 0 and centroid[1] > H // 2:
                    # down -> user is going in, plus delta by 1
                    delta += 1
                    totalDown += 1
                    to.counted = True
        trackableObjects[objectID] = to
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        coordinatex.append(centroid[0])
        coordinatey.append(centroid[1])
    #for loop
    for j in range(len(coordinatex)):
        for k in range(j+1,len(coordinatex)):
            distance = math.sqrt(pow(coordinatex[j]-coordinatex[k],2)+pow(coordinatey[j]-coordinatey[k],2))
            if distance < (0.1*coordinatey[j]+40):
                cv2.line(frame, (coordinatex[j],coordinatey[j]), (coordinatex[k],coordinatey[k]), (255,0,0), 3)
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if writer is not None:
        writer.write(frame)
    totalFrames += 1
    fps.update()

client = mqtt.Client()
client.connect("127.0.0.1", 1883, 60)
client.subscribe("camera")
client.on_message = on_message
client.loop_start()
set_interval(updateoccupancy,2)

def finish():
    fps.stop()
    print("\n[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    if writer is not None:
        writer.release()

print("[INFO] Ready for video input")
while True:
    S = input()
    if(S == "stop"):
        finish()
        client.disconnect()
        exit(0)