import numpy as np
import cv2
from PIL import Image
from rknn.api import RKNN
from timeit import default_timer as timer
import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
import ffmpeg
import math

CLASSES = ("person","hat","head","uniform_y","uniform_b","armour","cloth","work_trousers","normal_trousers")
GRID0 = 13
GRID1 = 26
LISTSIZE = len(CLASSES) +5
SPAN = 3
NUM_CLS = len(CLASSES)
MAX_BOXES = 500
OBJ_THRESH = 0.25
NMS_THRESH = 0.5

def load_model(rknn_model_path):
        rknn = RKNN()
        print('-->loading model')
        rknn.load_rknn(rknn_model_path)
        print('loading model done')

        print('--> Init runtime environment')
        
        # Init runtime environment
        print('--> Init runtime environment')
        #if platform.machine() == 'aarch64':
        #target = 'rk3399pro' 当前板子的NPU target是不用写的，写了就变成外接的rk3399pro了
        # http://t.rock-chips.com/forum.php?mod=viewthread&tid=1297
        target = None
        #else:
               # target = 'rk1808' 
        ret =rknn.init_runtime(target=target)

        #ret = rknn.init_runtime(target=None)
        if ret != 0:
                print('Init runtime environment failed')
                exit(ret)
        print('done')
        return rknn

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = input[..., 4]
    obj_thresh = -np.log(1/OBJ_THRESH - 1)
    pos = np.where(box_confidence > obj_thresh)
    input = input[pos]
    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    for idx, val in enumerate(pos[2]):
        box_wh[idx] = box_wh[idx] * anchors[pos[2][idx]]
    pos0 = np.array(pos[0])[:, np.newaxis]
    pos1 = np.array(pos[1])[:, np.newaxis]
    grid = np.concatenate((pos1, pos0), axis=1)
    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov4tiny_post_process(input_data):
    # # yolov3
    # masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
    #            [59, 119], [116, 90], [156, 198], [373, 326]]
    # yolov3-tiny
    masks = [[3, 4, 5], [0, 1, 2]]
    anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]


    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    # # Scale boxes back to original image shape.
    # width, height = 416, 416 #shape[1], shape[0]
    # image_dims = [width, height, width, height]
    # boxes = boxes * image_dims

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, scores,classes

def transform(image, boxes, scores, classes):
        labels=[]
        bboxs=[]
        confidents=[]
        for label, confidence, bbox in zip(classes, scores, boxes):
            label=CLASSES[label]
            x, y, w, h = bbox
            x *= image.shape[1]
            y *= image.shape[0]
            w *= image.shape[1]
            h *= image.shape[0]
            x=x+w/2
            y=y+h/2
            labels.append(label)
            bboxs.append([x,y,w,h])
            confidents.append(confidence)
        return bboxs,confidents,labels

def open_camera(camera_ip):
        camera=camera_ip
        probe = ffmpeg.probe(camera)
        print('probe ok')
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        out = (
            ffmpeg
                .input(camera, rtsp_transport='tcp')
                .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="quiet", r=10)
                .run_async(pipe_stdout=True)
        )
        return out,width,height

def rknn_model(image,rknn):
        out_boxes, out_boxes2 = rknn.inference(inputs=[image])
        out_boxes = out_boxes.reshape(SPAN, LISTSIZE, GRID0, GRID0)
        out_boxes2 = out_boxes2.reshape(SPAN, LISTSIZE, GRID1, GRID1)               
        input_data = []
        input_data.append(np.transpose(out_boxes, (2, 3, 0, 1)))
        input_data.append(np.transpose(out_boxes2, (2, 3, 0, 1)))
        boxes,scores,classes = yolov4tiny_post_process(input_data)
        return boxes,scores,classes

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        # print('class: {0}, score: {1:.2f}'.format(CLASSES[cl], score))
        # print('box coordinate x,y,w,h: {0}'.format(box))