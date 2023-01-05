import base64
import requests
import json
import cv2 as cv

url = 'http://127.0.0.1:5000/api'
def main():
    f = open('/home/lx/1.jpg', 'rb')
    #base64编码
    base64_data = base64.b64encode(f.read())
    f.close()
    '''注意编码类型问题，byte->string '''
    base64_data = base64_data.decode()
    #传输的数据格式
    data = {'img':base64_data}
    #post传递数据
    r = requests.post(url,data =json.dumps(data))
    return "OK"

if __name__ == '__main__':
    main()
