from aip import AipOcr

import cv2
import time
import numpy as np

import time
from loguru import logger
from codelab_adapter_client import AdapterNode
from threading import Thread


def img2txt_codelab(content,img2txt):

    #百度API通行证（获取方法见https://ai.baidu.com）
    APP_ID = '21223672'
    API_KEY = 'RrOZkIYFuMo6zHXf17mZ6Yko'
    SECRET_KEY = 'bFaIXMoP6Qzl2CQ3e57BKiKx3vo6Grfn'

    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

    """ 读取图片 """
    #img_path = 'D:\my_code\my_baidu.jpg'
    img_path = content
    raw = cv2.imread(img_path)
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()


    image = get_file_content(img_path)
    """ 调用通用文字识别, 图片参数为本地图片 """

    """ 如果有可选参数 """
    options = {}
    options["language_type"] = "CHN_ENG"
    options["detect_direction"] = "true"
    options["detect_language"] = "true"
    options["probability"] = "true"

    """ 带参数调用通用文字识别, 图片参数为本地图片 """
    res = client.basicGeneral(image, options)  # 返回识别结果
    #print(res)  # 打印
    num = res['words_result_num']
    words = []
    for i in range(num):
        words.append(res['words_result'][i]['words'])
    print(words)
    img2txt.send_message_to_scratch(words)  #将识别结果发送给scratch
    # for i in range(num):
    #     cv2.putText(raw, words[i],
    #                                 (20, 20*i+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                                 (255, 0, 0), 1)
    cv2.imshow('out',raw)
    cv2.waitKey()
    cv2.destroyAllWindows()


class EIMNode(AdapterNode):
    NODE_ID = "eim/img2txt"
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
            t= Thread(target=img2txt_codelab, args=(content,self,))
            t.start()
            t.join()

            # print(result["result"][0]['keyword'])
    def run(self):
        while self._running:
            time.sleep(1)



if __name__ == "__main__":
    try:
        node = EIMNode()
        node.receive_loop_as_thread()
        node.run()
    except KeyboardInterrupt:
        node.terminate()  # Clean up before exiting.