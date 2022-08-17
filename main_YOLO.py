import numpy as np
import cv2
from PIL import Image
from timeit import default_timer as timer
import argparse
import time
from rknn.api import RKNN
import ffmpeg
import json
import check_fire_cig_smoke
from rknn_fire import *
GRID0 = 13
GRID1 = 26
CLASSES = ("cigarette", "fire", "smoke")
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
for c2d in code2desc:
    code = c2d['AlgCode']
    desc = c2d['CodeDesc']
    if '_' in desc:
        cate2code[desc.split('_')[-1]] = code
rknn_model_path = './fire.rknn'
def parser():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--thresh", type=int)
    parser.add_argument("--AlgCode", type=int)
    parser.add_argument("--TaskId", type=str)
    parser.add_argument("--gap", type=float)
    parser.add_argument("--Desc", type=str)
    parser.add_argument("--alarm_pushUrl", type=str)
    parser.add_argument("--dev_code", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parser()
    show = True
    rknn = load_model(rknn_model_path)
    check = check_fire_cig_smoke.Check_fire()
    font = cv2.FONT_HERSHEY_SIMPLEX
    out, width, height = open_camera(camera_ip)
    accum_time = 0
    curr_fps = 0
    prev_time = timer()
    fps = "FPS: ??"
    try:
        cnt_empty = 0
        while (True):
            in_bytes = out.stdout.read(height * width * 3)
            if not in_bytes:
                cnt_empty += 1
                if cnt_empty > 20:
                    break
                continue
            cnt_empty = 0
            frame = np.frombuffer(in_bytes, dtype=np.uint8).reshape(height, width, 3)
            ret = True
            if ret == True:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (416, 416))
                #########################################################
                boxes, scores, classes = rknn_model(image, rknn)
                if boxes is not None:
                    boxes, scores, classes = transform(frame, boxes, scores, classes)
                    check.check(frame, boxes, classes, scores, './infos', args.AlgCode, args.TaskId, args.Desc,
                                args.gap, args.alarm_pushUrl, args.dev_code,cate2code)
                #########################################################
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time += exec_time
                curr_fps += 1
                if accum_time > 1:
                    accum_time -= 1
                    fps = "FPS: " + str(curr_fps)
                    tempfps  = curr_fps
                    curr_fps = 0
                cv2.putText(frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=0.50, color=(255, 0, 0), thickness=2)
                if show:
                    cv2.imshow("results", frame)
                    c = cv2.waitKey(tempfps) & 0xff
                    if c == 27:
                        cv2.destroyAllWindows()
                        rknn.release()
                        break
                testtime2=timer()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        rknn.release()
