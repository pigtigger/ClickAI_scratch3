from aip import  AipBodyAnalysis
import cv2
import time
import numpy as np

import time
from loguru import logger
from codelab_adapter_client import AdapterNode
from threading import Thread

def pose_codelab(content,pose):
    #百度API通行证（获取方法见https://ai.baidu.com）
    APP_ID = '21219492'
    API_KEY = 'jC6Z4Fa9jTs6wtpUTkoyGVMV'
    SECRET_KEY = 'pbGBVFu41ZZRWHhtPEqUbGTaEekaKfqc'
    client = AipBodyAnalysis(APP_ID,API_KEY,SECRET_KEY)  #输入ID和密钥建立连接
    video_capture = cv2.VideoCapture(0)
    while True:
        
        ret, frame = video_capture.read()#读取摄像头数据

        #处理图片格式并发送给云端API
        frame_b = cv2.imencode(".jpg", frame)[1].tobytes()
        gesture_info = client.gesture(frame_b)

        #处理识别结果（API会将识别到的人脸和手势全部返回）
        if gesture_info['result']:
            result = gesture_info['result']
            num = gesture_info['result_num']
            for i in range(num):

                classname = result[i]['classname']#识别的手势/人脸类别

                #识别框坐标
                left = result[i]['left']
                top = result[i]['top']
                width = result[i]['width']
                height = result[i]['height']

                #绘制识别框
                cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 0, 255), 2)
                if (classname == 'Face'):
                    cv2.putText(frame, 'Face',
                                (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 1)
                else:
                    cv2.putText(frame, '{}'.format(classname),
                                (left, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)
                    pose.send_message_to_scratch(classname)#发送识别结果到scratch

        cv2.imshow('Video', frame)

        time.sleep(0.15)#防止QPS溢出

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # time.sleep(5)
    #
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

class EIMNode(AdapterNode):
    NODE_ID = "eim/pose"
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

        if type(content) == str :
            t= Thread(target=pose_codelab, args=(content,self,))
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