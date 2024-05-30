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
                    ih, iw, ic = img.shape
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
        self.cap.set(640, width)
        self.cap.set(480, height)
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
resize_Height = 600

webcam_thread1 = WebcamThread(0, resize_Width, resize_Height)
webcam_thread2 = WebcamThread(1, resize_Width, resize_Height)
webcam_thread1.start()
webcam_thread2.start()

segmentor = SelfiSegmentation()

#------------ NDI -----------------------
send_settings = ndi.SendCreate()
send_settings.ndi_name = 'python_open_CV'
ndi_send = ndi.send_create(send_settings)
video_frame = ndi.VideoFrameV2()
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    if webcam_thread1.img is not None and webcam_thread2.img is not None:
        
        img1 = webcam_thread1.img
        img2 = webcam_thread2.img
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)

        ndi_img = cvzone.stackImages([img2, img1], 2 , 1)
        faceimg1 = cvzone.stackImages([img2, img1], 2 , 1)
        face_img, bboxs = detector.findFaces(faceimg1, draw=False)
        if bboxs:
            for bbox in bboxs:
                # ---- Get Data  ---- #
                center = bbox["center"]
                x, y, w, h = bbox['bbox']

                # ---- Draw Data  ---- #
                cv2.circle(face_img, center, 5, (255, 0, 255), cv2.FILLED)
                cvzone.cornerRect(face_img, (x, y, w, h))
                print("0",x, "1", y,"3",w,"4",h)
                
        ndi_send_img = segmentor.removeBG(ndi_img, (0,0,0), cutThreshold=0.7)
        cut_ndi_send_img = ndi_send_img[0:540, 0:960]

        showImage = cvzone.stackImages([face_img, ndi_send_img], 2 , 1)

        video_frame.data = cut_ndi_send_img
        video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
        ndi.send_send_video_v2(ndi_send, video_frame)
        outImage = cv2.resize(showImage, (1060, 370))
        cv2.imshow("Output", outImage)
        

    key = cv2.waitKey(1)
    if key == 27:
        break

    time.sleep(1/ 60)

webcam_thread1.stop()
webcam_thread2.stop()
ndi.send_destroy(ndi_send)
ndi.destroy()
cv2.destroyAllWindows()