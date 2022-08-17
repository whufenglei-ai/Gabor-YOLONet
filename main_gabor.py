import numpy as np
import cv2
from PIL import Image
from timeit import default_timer as timer
import argparse
import time
from rknn.api import RKNN
import ffmpeg
import json
from check_hat_cass import Check_hat
from rknn_hatArmour import *
CLASSES = ("hat","person")
GRID0 = 13
GRID1 = 26
LISTSIZE = len(CLASSES) +5
SPAN = 3
NUM_CLS = len(CLASSES)
MAX_BOXES = 500
OBJ_THRESH = 0.25
NMS_THRESH = 0.5

#布控球ip
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

rknn_model_path='./hat_person.rknn'
rknn_class_model_path = './armour_class.rknn'

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
    rknn_class = load_model(rknn_class_model_path)
    check = Check_hat()
    font = cv2.FONT_HERSHEY_SIMPLEX
    out,width,height=open_camera(camera_ip)
    accum_time = 0
    curr_fps = 0
    prev_time = timer()
    fps = "FPS: ??"
    try:
        cnt_empty = 0
        cnt = 0
        while(True):
            cnt+=1
            if cnt%2==0:continue
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
                boxes,scores,classes = rknn_model(image,rknn)
                if boxes is not None:
                    boxes, scores, classes = transform(frame, boxes, scores, classes)
                    person_bbox, clear_img = check.check(frame,boxes,classes,scores,'./infos', args.AlgCode, args.TaskId,args.Desc, args.gap, args.alarm_pushUrl, args.dev_code,cate2code)
                    is_armour = True
                    for pbox in person_bbox:
                        left, top, right, bot = pbox
                        person_img = clear_img[top:bot, left:right, :]
                        person_img = cv2.resize(person_img, (224, 224))        
                        outpp = rknn_class.inference(inputs=[person_img])
                        yes = float(outpp[0][0][0])
                        no = float(outpp[0][0][1])
                        cate = 'yes' if yes>0.4 else 'no'

                        if cate == 'no':
                            is_armour = False
                            if check.err_count['armour']>=3:
                                cv2.rectangle(frame,(left,top),(right,bot),color=(0,0,255),thickness=1)
                                cv2.putText(frame, 'no armour',(left, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                            else:
                                cv2.putText(image, 'armour', (left, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if not is_armour:
                        if check.err_count['armour']>=30:
                            check.threadPool.submit(check.sendRequest, image, cate2code['armour'], args.TaskId,args. Desc, args.gap, args.alarm_pushUrl, args.dev_code, 'armour')
                            check.err_count['armour'] = 0
                        else:
                            check.err_count['armour'] += 1
                    else:
                        if check.err_count['armour']>0:
                            check.err_count['armour']-=5
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
