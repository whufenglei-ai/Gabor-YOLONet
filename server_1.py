from flask import Flask, request, jsonify
import json
from gevent import pywsgi
import cv2
# from settings import APP_PORT
import os
# 创建一个服务
app = Flask(__name__)
import uuid

# 创建一个接口 指定路由和请求方法 定义处理请求的函数
@app.route(rule='/analysis/alarm_push/', methods=['POST'])
def everything():
    print('request.data:', request.data)
    save_info_file = f'./save/{uuid.uuid4().hex}.txt'
    if request.files:
        request.files['image'].save(os.path.join('./save',request.files['image'].filename))
    else:
        with open(save_info_file, 'w') as f:
            f.write(request.data.decode('utf-8'))
    return request.data
if __name__ == '__main__':
    server = pywsgi.WSGIServer(('127.0.0.1', 8807), app)
    print('server is running...')
    server.serve_forever()

