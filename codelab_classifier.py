import time
from loguru import logger
from codelab_adapter_client import AdapterNode
import numpy as np
import cv2
from aip import AipImageClassify
from threading import Thread

#百度API通行证（获取方法见https://ai.baidu.com）
APP_ID = '19037351'
API_KEY = 'jm1Gn3KDwcXqCKfbbE2LtEK2'
SECRET_KEY = 'YRGKmItwOPkep3MPpenb0EmxUD2GDWfo'

#输入ID和密钥建立连接
client = AipImageClassify(APP_ID, API_KEY, SECRET_KEY)

#定义文件读取函数
def get_file_content(filePath):
    with open(filePath,'rb') as fp:
        return fp.read()

#EIM Node
class EIMNode(AdapterNode):
    NODE_ID = "eim/classifier"
    DESCRIPTION = "Everything Is a Message"
    HELP_URL = "https://adapter.codelab.club/extension_guide/eim/"
    def __init__(self):
        super().__init__()

    def send_message_to_scratch(self, content):
        message = self.message_template()
        message["payload"]["content"] = content
        self.publish(message)

    def extension_message_handle(self, topic, payload):
        self.logger.info(f'the message payload from scratch: {payload}')
        content = payload["content"]
        print(topic)
        print(content)
        if type(content) == str:
            t= Thread(target=classifier, args=(content,self,))
            t.start()
            t.join()

            # print(result["result"][0]['keyword'])
    def run(self):
        while self._running:
            time.sleep(1)

#定义图片分类函数
def classifier(content,classify):

    image = get_file_content(content)#根据content路径读取图片
    img = cv2.imread(content)
    result = client.advancedGeneral(image)#调用云端API返回结果
    content_send_to_scratch = result["result"][0]['keyword']
    classify.send_message_to_scratch(content_send_to_scratch)#向scratch发送结果

    #显示图片
    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        node = EIMNode()
        node.receive_loop_as_thread()
        node.run()
    except KeyboardInterrupt:
        node.terminate()  # Clean up before exiting.

