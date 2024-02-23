from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import os
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

warnings.filterwarnings('ignore')

def write_txt(name, msg):
    path_name = 'D:\Project\Pycharm\deep_sort_yolov3-master'
    full_path_name = path_name + name + '.txt'
    if not os.path.exists(full_path_name):
        file = open(full_path_name, 'w')
    file = open(full_path_name, 'a')
    file.write(msg)

def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True

    video_capture = cv2.VideoCapture(r'D:\Desktop\导航\样本\视频\man1_short_2.mp4')

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3)) #在视频流的帧的宽度
        h = int(video_capture.get(4)) #在视频流的帧的高度
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') #设置需要保存视频的格式
        out = cv2.VideoWriter('output001.avi', fourcc, 15, (w, h))

    fps = 0.0

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        t1 = time.time() #返回当前时间的时间戳

        image = Image.fromarray(frame[..., ::-1])  #实现array到image的转换  # bgr to rgb
        boxs = yolo.detect_image(image)[0]
        confidence = yolo.detect_image(image)[1]

        features = encoder(frame, boxs) #以encoding指定的编码格式编码字符串

        detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in
                      zip(boxs, confidence, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # 追踪框信息
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

        # 检测框信息
        # 这里是描述框子的输出信息
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)
            center_x = int((int(bbox[0]) + int(bbox[2])) / 2)
            center_y = int((int(bbox[1]) + int(bbox[3])) / 2)
            point = ' [' + str(center_x) + ',' + str(center_y) + ']'
            txt = point
            cv2.putText(frame, txt, (int(bbox[0]), int(bbox[1])), 0, 0.5, (255, 255, 255), 1)
            points_lefttop_txt=str(int(bbox[0]))+',' + str(int(bbox[2]))
            points_rightbuttom_txt=str(int(bbox[1]))+',' + str(int(bbox[3]))
            write_txt('\points_lefttop', points_lefttop_txt + '\n')
            write_txt('\points_rightbuttom', points_rightbuttom_txt + '\n')

            weight=int(bbox[2])-int(bbox[0])
            length=int(bbox[1])-int(bbox[3])
            points_w_l_txt=str(weight) + ',' +str(length)
            write_txt('\points_w_l', points_w_l_txt + '\n')
            print(point)


        cv2.imshow('', frame)

        # fps = (fps + (1. / (time.time() - t1))) / 2
        # print("FPS = %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
