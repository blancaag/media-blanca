import os
import sys
import pkg_resources

import numpy as np
import tensorflow as tf
from document_check_framework.monitoring import logger

from ..train_models.MTCNN_config import config

log = logger.get_logger(__name__)


class FcnDetector(object):
    #net_factory: which net
    #model_path: where the params'file is
    def __init__(self, net_factory, model_path):
        #create a graph
        graph = tf.Graph()
        with graph.as_default():
            #define tensor and op in graph(-1,1)
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op,
                                       [1, self.height_op, self.width_op, 3])
            #self.cls_prob batch*2
            #self.bbox_pred batch*4
            #construct model here
            #self.cls_prob, self.bbox_pred = net_factory(image_reshape, training=False)
            #contains landmark
            self.cls_prob, self.bbox_pred, _ = net_factory(
                image_reshape, training=False)

            #allow
            self.sess = tf.Session(
                config=tf.ConfigProto(
                    allow_soft_placement=True,
                    gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid

            # Getting the location of the model.
            parts = model_path.split('/')  # e.g.: mtcnn, dep, PNet, PNet-18
            # Since we bundle this as a package, we need to use `pkg_resources`
            # The path to the model has to be of the form (package, resource)

            # We first need the model directory.
            package = '.'.join(parts[:-2])  # e.g.: mtcnn.dependencies
            resource = parts[-2]  # e.g.: ONet
            ckpt_dir = pkg_resources.resource_filename(package, resource)

            # Then the path to the model.
            package = '.'.join(parts[:-1])  # e.g.: mtcnn.dependencies.ONet
            resource = parts[-1]  # e.g.: ONet-18
            model_path = pkg_resources.resource_filename(package, resource)

            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            #import ipdb; ipdb.set_trace()
            readstate = ckpt and ckpt.model_checkpoint_path
            assert readstate, "the params dictionary is not valid"
            log.debug('Restoring models parameters')
            saver.restore(self.sess, model_path)

    def predict(self, databatch):
        height, width, _ = databatch.shape
        cls_prob, bbox_pred = self.sess.run(
            [self.cls_prob, self.bbox_pred],
            feed_dict={
                self.image_op: databatch,
                self.width_op: width,
                self.height_op: height
            })
        return cls_prob, bbox_pred
