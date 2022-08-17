from check_fence import Check_fence
from rknn_fence import *
from threading import Thread
from queue import Queue
import numpy as np
import cv2
from PIL import Image
from timeit import default_timer as timer
import argparse
import time
from rknn.api import RKNN
import ffmpeg
import json


GRID0 = 13
GRID1 = 26
CLASSES = ("person","fence")
SPAN = 3
NUM_CLS = len(CLASSES)
MAX_BOXES = 500
OBJ_THRESH = 0.25
NMS_THRESH = 0.5
# 布控球ip
with open('algCode.json', 'r') as f:
    configs = json.load(f)
camera_ip = configs['camera_ip']
code2desc = configs['code2desc']
cate2code = {}
cate2desc = {}
for c2d in code2desc:
    code = c2d['AlgCode']
    desc = c2d['CodeDesc']
    if '_' in desc:
        cate = desc.split('_')[-1]
        cate2code[cate] = code
        cate2desc[cate] = desc

rknn_model_path='./fence322.rknn'


def parser():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--thresh", type=int)
    parser.add_argument("--AlgCode", type=str)
    parser.add_argument("--TaskId", type=str)
    parser.add_argument("--gap", type=float)
    parser.add_argument("--Desc", type=str)
    parser.add_argument("--alarm_pushUrl", type=str)
    parser.add_argument("--dev_code", type=str)
    return parser.parse_args()


def video_capture(frame_queue, check_image_queue):
    while flag:
        ret,frame = video.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_resize = cv2.resize(image, (416, 416))
            frame_queue.put(frame)
            check_image_queue.put(image_resize)


def inference(check_image_queue, detections_queue, prev_time_queue):
    while flag:
        check_image = check_image_queue.get()
        prev_time_queue.put(time.time())
        boxes, scores, classes = rknn_model(check_image, rknn)
        detections_queue.put((boxes, scores, classes))


def drawing(frame_queue, detections_queue, prev_time_queue):
    global flag
    while flag:
        frame = frame_queue.get()
        detections = detections_queue.get()
        prev_time = prev_time_queue.get()
        if detections[0] is not None:
            boxes, scores, classes = detections
            boxes, scores, classes = transform(frame, boxes, scores, classes)
            check.check(frame, boxes, classes, scores, './infos', args.AlgCode, args.TaskId,cate2desc,
                        args.gap, args.alarm_pushUrl, args.dev_code,cate2code)
        fps = f'FPS: {int(1 / (time.time() - prev_time))}'
        cv2.putText(frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        if show:
            cv2.imshow('Inference', frame)
            if cv2.waitKey(int(1 / (time.time() - prev_time))) == 27:
                flag = False
                break
args = parser()
flag = True
show = True
rknn = load_model(rknn_model_path)
check = Check_fence()
font = cv2.FONT_HERSHEY_SIMPLEX
video = cv2.VideoCapture('fence_2.mp4')

frame_queue = Queue()
check_image_queue = Queue(maxsize=1)
detections_queue = Queue(maxsize=1)
prev_time_queue = Queue(maxsize=1)

Thread(target=video_capture, args=(frame_queue, check_image_queue)).start()
Thread(target=inference, args=(check_image_queue, detections_queue, prev_time_queue)).start()
Thread(target=drawing, args=(frame_queue, detections_queue, prev_time_queue)).start()

