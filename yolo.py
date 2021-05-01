# -------------------------------------#
#       YOLO Model
# -------------------------------------#
import colorsys
import os
import winsound
from playsound import playsound
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from nets.yolo4 import YoloBody
from utils.utils import (DecodeBox, bbox_iou, letterbox_image,
                         non_max_suppression, yolo_correct_boxes)

# ----------------------------------------------------------#
#   Needs to be modified the model_path and classes_path
#   Choosing the epoch by comparing the loss value
# ----------------------------------------------------------#
class YOLO(object):
    _defaults = {
        "model_path": 'logs/Epoch46-Total_Loss0.9438-Val_Loss1.6428.pth',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "model_image_size": (416, 416, 3),
        "confidence": 0.5,
        "iou": 0.3,
        "cuda": True,
        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #    Initialize YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    # ---------------------------------------------------#
    #   Get all classes
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   Get all anchor
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    # ---------------------------------------------------#
    #   Build up model
    # ---------------------------------------------------#
    def generate(self):
        # ---------------------------------------------------#
        #   YOLOv4 Model
        # ---------------------------------------------------#
        self.net = YoloBody(len(self.anchors[0]), len(self.class_names)).eval()

        # ---------------------------------------------------#
        #   Load YOLOv4 model weights
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        print('Finished!')

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        # ------------------------------------------------------#
        #   The specific decoding process (call DecodeBox class)
        # ------------------------------------------------------#
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], len(self.class_names), (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # Set different colors for the frame
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    # ---------------------------------------------------#
    #   Image detection
    # ---------------------------------------------------#
    def detect_image(self, image, coAlarm, rtAlarm):
        image_shape = np.array(np.shape(image)[0:2])

        # -------------------------------------------------------------#
        #   Keep the ratio of w/h, so that the object would not deform
        # -------------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1], self.model_image_size[0])))
        else:
            crop_img = image.convert('RGB')
            crop_img = crop_img.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        photo = np.array(crop_img, dtype=np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------------#
            #   Image passes to the net
            # ---------------------------------------------------------#
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))

            # ---------------------------------------------------------#
            #  Non-maximum suppression non_max_suppression
            # ---------------------------------------------------------#
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, len(self.class_names),
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.iou)

            # ---------------------------------------------------------#
            #   Return image if no object was detected
            # ---------------------------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            top_index = batch_detections[:, 4] * batch_detections[:, 5] > self.confidence
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            top_label = np.array(batch_detections[top_index, -1], np.int32)
            top_bboxes = np.array(batch_detections[top_index, :4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:, 0], -1), np.expand_dims(
                top_bboxes[:, 1], -1), np.expand_dims(top_bboxes[:, 2], -1), np.expand_dims(top_bboxes[:, 3], -1)

            # -----------------------------------------------------------------#
            #   Remove gray bars
            # -----------------------------------------------------------------#
            if self.letterbox_image:
                boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                           np.array([self.model_image_size[0], self.model_image_size[1]]), image_shape)
            else:
                top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                boxes = np.concatenate([top_ymin, top_xmin, top_ymax, top_xmax], axis=-1)

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        bedLeft, bedRight, bedBottom, patientLeft, patientRight, patientBottom = (0, 0, 0, 0, 0, 0)
        climbOut = 0
        objects = 0

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # class 0 = patient
            if predicted_class == "patient":
                # patientLeft, patientRight, patientBottom = (0, 0, 0)
                patientLeft = left
                patientRight = right
                patientBottom = bottom
                objects += 1

            # class 1 = bed
            if predicted_class == "bed":
                # bedLeft, bedRight, bedBottom = (0, 0, 0)
                bedLeft = left
                bedRight = right
                bedBottom = bottom
                bedTop = top
                objects += 1

            # check only if bed and patient are exist
            if objects == 2:
                # 3 condition

                if coAlarm == "yes":
                    if (patientLeft < bedLeft) or (patientRight > bedRight) or (patientBottom > bedBottom):
                        climbOut = 1
                        # playsound('1.mp3')
                        winsound.Beep(32767, 100)

                    elif (0.1 > float(patientLeft - bedLeft) / float(bedRight - bedLeft) or
                          0.1 > float(bedRight - patientRight) / float(bedRight - bedLeft) or
                          0.1 > float(patientBottom - bedBottom) / float(bedTop - bedBottom)):
                        # alarming
                        # frequency is set to 500Hz
                        freq = 500
                        # duration is set to 100 milliseconds
                        dur = 100
                        winsound.Beep(freq, dur)

            if predicted_class == "rmTube":
                if rtAlarm == "yes":
                    winsound.Beep(32767, 100)

            # Picture frame
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print coordinates
            # print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image, climbOut
