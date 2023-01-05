from flask import Flask, request
import cv2
import numpy as np
import base64
import json

app = Flask(__name__)

'''
路由器端口api，传输方式POST
GET用于从服务器端获取数据，包括静态资源(HTML|JS|CSS|Image等等)、动态数据展示(列表数据、详情数据等等)。
POST用于向服务器提交数据，比如增删改数据，提交一个表单新建一个用户、或修改一个用户等。
'''
@app.route('/api',methods=['POST'])

def api():
    #来捕捉客户端传来的数据
    data = request.get_data().decode('utf-8')
    #json.loads将string转换为dict
    data = json.loads(data)
    #获取dict中'img'标签的数据
    image_b64 = data["img"]
    #进行base64解码工作 base64->数组
    image_decode = base64.b64decode(image_b64)
    #fromstring实现了字符串到Ascii码的转换
    nparr = np.fromstring(image_decode, np.uint8)
    #从nparr中读取数据，并把数据转换(解码)成图像格式
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('./demo/test.jpg',img_np)
    return 'OK'

if __name__ == '__main__':
    app.run(debug=True)