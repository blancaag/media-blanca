#coding:utf-8
import os
import sys

import cv2
import numpy as np

from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
from . import settings


# We want to use this project as a package. We'll make several calls
# to `detect_factes` in one python session. This is ineffective because:
# - at each call, we create a new session, which is time consuming
# - at each call, the previous session is *not* destroyed. Hence a memory
# leak.
# A trivial (and maybe non-optimal) way to fix this is to use global variables
# that will be stored throughout the execution.
# We'll use only one and only one session, avoiding the pitfalls described
# earlier.
global PNet
global ONet
global RNet
PNet = None
ONet = None
RNet = None


def detect_faces(images, min_face_size=24.,
                 thresholds=None,
                 stride=2, slide_window=False,
                 test_mode='ONet'):
    """

    Args:
        images (list of np.ndarray): the images to be processed, in RGB.
    """
    if not isinstance(images, list):
        # We assume that `images` is actually a single image
        images = [images]
    if thresholds is None:
        thresholds = [settings.PNET_SETTINGS['threshold'],
                      settings.RNET_SETTINGS['threshold'],
                      settings.ONET_SETTINGS['threshold']]

    global PNet
    global ONet
    global RNet
    detectors = {}

    if PNet is None:
        # load pnet model
        if slide_window:
            PNet = Detector(P_Net, 12,
                            settings.PNET_SETTINGS['batch_size'],
                            settings.PNET_SETTINGS['path'])
        else:
            PNet = FcnDetector(P_Net, settings.PNET_SETTINGS['path'])
    detectors['PNet'] = PNet

    if test_mode in ['RNet', 'ONet']:
        if RNet is None:
            # load rnet model
            RNet = Detector(R_Net, 24,
                            settings.RNET_SETTINGS['batch_size'],
                            settings.RNET_SETTINGS['path'])
        detectors['RNet'] = RNet

    if test_mode == 'ONet':
        if ONet is None:
            # load onet model
            ONet = Detector(O_Net, 48,
                            settings.ONET_SETTINGS['batch_size'],
                            settings.ONET_SETTINGS['path'])
        detectors['ONet'] = ONet

    mtcnn_detector = MtcnnDetector(
        detectors=detectors,
        min_face_size=min_face_size,
        stride=stride,
        threshold=thresholds,
        slide_window=slide_window)

    test_data = TestLoader(images)
    all_boxes, landmarks = mtcnn_detector.detect_face(test_data)

    # If we sent only 1 image, we don't expect a list of results. Just 1 rsult.
    if len(all_boxes) == 1:
        all_boxes = all_boxes[0]
        landmarks = landmarks[0]

    return all_boxes, landmarks
