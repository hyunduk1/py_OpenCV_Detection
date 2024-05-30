import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import NDIlib as ndi
import threading
import time
import pystray
import mediapipe as mp
from pystray import MenuItem
from pystray import Menu
import matplotlib.pyplot as plt
import FilterManager as filter
from pythonosc import udp_client
#import mic

ip = "192.168.0.133"
port = 2222
client = udp_client.SimpleUDPClient(ip, port)

# 차트 초기화
plt.figure(figsize=(10, 6))
plt.title('Pixel Values at y=150')
plt.xlabel('x')
plt.ylabel('Pixel Value')
plt.ion()

class FaceDetector:
    def __init__(self, minDetectionCon=0.5, modelSelection=0):
        self.minDetectionCon = minDetectionCon
        self.modelSelection = modelSelection
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence=self.minDetectionCon,
                                                                model_selection=self.modelSelection)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                if detection.score[0] > self.minDetectionCon:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw = img.shape[:2]  # 수정된 부분
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                    cx, cy = bbox[0] + (bbox[2] // 2), \
                             bbox[1] + (bbox[3] // 2)
                    bboxInfo = {"id": id, "bbox": bbox, "score": detection.score, "center": (cx, cy)}
                    bboxs.append(bboxInfo)
        return img, bboxs

# 해파리 카메라 해상도 640, 480 기준점 
class WebcamThread(threading.Thread): 
    def __init__(self, cam_id, width, height):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(1280, width)
        self.cap.set(960, height)
        self.cap.set(cv2.CAP_PROP_FPS, 100)
        self.running = False
        self.img = None

    def run(self):
        self.running = True
        while self.running:
            success, img = self.cap.read()
            if not success:
                break
            self.img = img

    def stop(self):
        self.running = False
        self.join()
        self.cap.release()



cut_start_Width = 200
cut_start_Height = 0
cut_Width = 300
cut_Height = 600
resize_Width = 300
resize_Height = 300

webcam_thread1 = WebcamThread(0, resize_Width, resize_Height)
webcam_thread1.start()

segmentor = SelfiSegmentation()

Is_Human_Enable = False

Index_Current_Enable_Num = 0
Index_Current_Disable_Num = 0



last_detected_time = time.time()
last_undetected_time = time.time()

#------------ NDI -----------------------
send_settings = ndi.SendCreate()
send_settings.ndi_name = 'python_open_CV'
ndi_send = ndi.send_create(send_settings)
video_frame = ndi.VideoFrameV2()
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    if webcam_thread1.img is not None:
        current_time = time.time()
        img1 = webcam_thread1.img
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.flip(img1, 1)
        #인풋값 필터 치는곳------------------------
        #img1 = filter.MedianFilter(img1, 5)

        # test_cvt = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        
        #----------------------------------


        img = cvzone.stackImages([img1], 1,1)
        ndi_img = img1
        
        face_img, bboxs = detector.findFaces(img, draw=False)
        if bboxs:
            for bbox in bboxs:
                center = bbox["center"]
                x, y, w, h = bbox['bbox']

                cv2.circle(face_img, center, 5, (255, 0, 255), cv2.FILLED)
                cvzone.cornerRect(face_img, (x, y, w, h))
                Index_Current_Disable_Num = 0
            #사람이 감지가 되었을때
            if Is_Human_Enable == True:
                Index_Current_Enable_Num += 1
                print("감지 숫자 : ",Index_Current_Enable_Num)
                if Index_Current_Enable_Num >= 10:
                    print("사람 감지")
                    client.send_message("/MSG_Video", 1)
                    Index_Current_Enable_Num = 0
                    Is_Human_Enable = False

        else:
            Index_Current_Enable_Num = 0
            #사람이 감지가 안되었을때
            if Is_Human_Enable == False:
                Index_Current_Enable_Num = 0
                Index_Current_Disable_Num += 1
                print("감지 안됌 : ", Index_Current_Disable_Num)
                if Index_Current_Disable_Num >= 10:
                    print("사람 없음")
                    client.send_message("/MSG_Video", 0)
                    Index_Current_Disable_Num = 0
                    Is_Human_Enable = True
            

        ndi_send_img = segmentor.removeBG(cv2.cvtColor(ndi_img, cv2.COLOR_GRAY2BGR), (0,0,0), cutThreshold=0.6)
        test_img = cv2.GaussianBlur(ndi_send_img,(3,3), 0)

        #아웃풋 필터 치는곳-----------------
        ndi_send_img = filter.MedianFilter(ndi_send_img, 3)
        #---------------------------------
        cut_ndi_send_img = ndi_send_img[0:640, 0:480]

        

        showImage = cvzone.stackImages([img, ndi_send_img], 2 , 1)

        video_frame.data = cut_ndi_send_img
        video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
        ndi.send_send_video_v2(ndi_send, video_frame)
        cv2.imshow("Output", showImage)

        #region 차트 그리기 (캘리브레이션)------------
        row_values = ndi_send_img[300, :, 0]
        threshold_value = 100

        plt.clf()
        plt.plot(row_values, color='b', lw=2)
        plt.axhline(y=threshold_value, color='r', linestyle='--', label='Threshold Value')

        plt.pause(0.01)
        

    key = cv2.waitKey(1)
    if key == 27:
        break

    time.sleep(1/ 60)


plt.ioff()
plt.close()
webcam_thread1.stop()
ndi.send_destroy(ndi_send)
ndi.destroy()
cv2.destroyAllWindows()