#!/usr/bin/env python3

import rospy
import cv2
import os
import sys
import torch
import numpy as np

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Int64  
from scale_car_yolov5.msg import Yolo_Objects, Objects
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

count = 0


class YoloV5_ROS():
    def __init__(self):
        rospy.Subscriber(source, CompressedImage, self.Callback)
        self.pub = rospy.Publisher("yolov5_pub", data_class=Yolo_Objects, queue_size=10)
        self.weights = rospy.get_param("~weights")
        self.data = rospy.get_param("~data")
        self.device = torch.device(rospy.get_param("~device"))

        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = (640, 640)

        self.conf_thres = 0.85
        self.iou_thres = 0.45
        self.max_det = 10
        self.classes = None
        self.agnostic_nms = False
        self.line_thickness = 3

        self.hide_labels = False
        self.hide_conf = False

        self.area = 0

    def Callback(self, data):
        global count
        bridge = CvBridge()
        img = bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        msg = Yolo_Objects()

        cv2.namedWindow('result', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        if count % 1 == 0:
            im0s = img
            self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))

            im = letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()
            im /= 255

            if len(im.shape) == 3:
                im = im[None]

            pred = self.model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)


            for i, det in enumerate(pred):
                gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
                annotator = Annotator(im0s, line_width=self.line_thickness, example=str(self.names))

                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    x1, y1, x2, y2 = map(int, xyxy)
                    area = (x2 - x1) * (y2 - y1)

                    msg = Yolo_Objects()
                    msg.yolo_objects.append(Objects(c, x1, x2, y1, y2))

                    for obj in msg.yolo_objects:
                        msg = obj.c
                        if area >= 35000:
                            self.pub.publish(msg)
                            print(f"{msg}, {area}")
                        else:
                            pass

            cv2.imshow('result', im0s)
            cv2.waitKey(1)

        


def run():
    global source
    rospy.init_node("yolov5_ros")
    source = rospy.get_param("~source")
    detect = YoloV5_ROS()
    rospy.spin()


if __name__ == '__main__':
    run()
