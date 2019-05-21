import os
import logging
import json

import numpy as np
import tensorflow as tf
import cv2

from sfmt.endernewton.model.config import cfg
from sfmt.endernewton.model.test import im_detect
from sfmt.endernewton.model.nms_wrapper import nms
from sfmt.endernewton.nets.vgg16 import vgg16
from sfmt.endernewton.nets.resnet_v1 import resnetv1
from sfmt.endernewton.nets.mobilenet_v1 import mobilenetv1


#################################################################################
#
#       OBJECT DETECTOR BASE CLASS
#
#################################################################################


class _BaseDetector():
    """Base interface for different types of object detectors.
    The parameter box_size_threshold is a tuple with two values,
    specifying the maximum width and height of bounding boxes. Larger boxes are
    filtered out, if this parameter is not None. If not None the scaling factor
    determines rescaling of bounding box coordinates. This is useful if the input
    frame size of the detector differs from the frame size of further processing
    steps.
    """
    def __init__(self, path, box_size_threshold, scaling_factor):
        self.path = path
        self.box_size_threshold = box_size_threshold
        self.scaling_factor = scaling_factor
        self._load_label_map()


    def _load_label_map(self):
        """Load the map for class labels from labelmap.json file."""
        self.label_map = []
        model_dir = os.path.dirname(self.path)
        label_map_file = os.path.join(model_dir, "labelmap.json")
        if os.path.isfile(label_map_file):
            self.label_map = json.load(open(label_map_file, "r"))


    def _class_index_to_label(self, class_index):
        """Maps a class index (`int`) to a class string."""
        if len(self.label_map) == 0:
            raise RuntimeError(("No label map was loaded. Either this model "
                "does not support label mapping or the label map file could "
                "not be loaded."))
        else:
            class_label = [map["label"] for map in self.label_map if map["index"] == class_index][0]
        return class_label


    def _get_output_dict(self, frame):
        """Get output dictionary of detector for a given frame."""
        raise NotImplementedError


    def _rescale_bounding_boxes(self, output_dict):
        """Rescale bounding box coordinates."""
        output_dict['detection_boxes'] = np.multiply(output_dict['detection_boxes'], self.scaling_factor)
        return output_dict


    def _filter_oversized_bounding_boxes(self, output_dict):
        """Filters out unnaturally large bounding boxes."""
        new_output_dict = {}
        new_output_dict['num_detections'] = 0
        new_output_dict['detection_boxes'] = []
        new_output_dict['detection_classes'] = []
        new_output_dict['detection_scores'] = []
        # fill entries into new dict, if the box size does not exceed width and height thresholds
        for i, bounding_box in enumerate(output_dict['detection_boxes']):
            if bounding_box[2] < self.box_size_threshold[0] and bounding_box[3] < self.box_size_threshold[1]:
                new_output_dict['detection_boxes'].append(output_dict['detection_boxes'][i])
                new_output_dict['detection_classes'].append(output_dict['detection_classes'][i])
                new_output_dict['detection_scores'].append(output_dict['detection_scores'][i])
                new_output_dict['num_detections'] = new_output_dict['num_detections'] + 1
       # convert to numpy arrays
        new_output_dict['detection_boxes'] = np.array(new_output_dict['detection_boxes'])
        new_output_dict['detection_classes'] = np.array(new_output_dict['detection_classes'])
        new_output_dict['detection_scores'] = np.array(new_output_dict['detection_scores'])
        return new_output_dict


    def detect(self, frame):
        """Run inference in object detection model.
        Function takes the frame in which to detect people and objects. Returns an
        output dict with keys 'num_detections', 'detection_classes', 'detection_boxes'
        and 'detection_scores'. Skips inference and returns an empty output dict if
        frame is None.
        """
        if frame is not None:
            output_dict =  self._get_output_dict(frame)
            # filter out too large bounding boxes
            if self.box_size_threshold is not None:
                output_dict = self._filter_oversized_bounding_boxes(output_dict)
            # rescale bounding box coordinates according to scaling factor
            if self.scaling_factor is not None:
                output_dict = self._rescale_bounding_boxes(output_dict)
        else:
            output_dict = {
                'num_detections': 0,
                'detection_boxes': np.empty(shape=(0, 4), dtype=np.float32),
                'detection_scores': np.empty(shape=(0,), dtype=np.float32),
                'detection_classes': np.empty(shape=(0,), dtype=np.str_)
            }
        return output_dict


    def draw_bounding_boxes(self, frame, output_dict, color=(0, 255, 0)):
        """Draws detector results on the frame.
        Draws bounding boxes as stored in output dict in specified color on the frame. Additionally,
        scores and object classes are shown. Size of the frame should be the same as the frame fed
        into detect method.
        """
        cv2.putText(frame, 'Detections: {}'.format(output_dict['num_detections']), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        for box, cls, score in zip(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores']):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[0] + box[2])
            ymax = int(box[1] + box[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, '{}'.format(cls), (xmin, ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, '{:1.3f}'.format(score), (xmin, ymax-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


#################################################################################
#
#       OBJECT DETECTOR OF TENSORFLOW OBJECT DETECTION API
#
#################################################################################


class DetectorTF(_BaseDetector):
    """Interface for object detectors of the Tensorflow Object Detection API.
    GitHub: [https://github.com/tensorflow/models/tree/master/research/object_detection]
    """

    def __init__(self, path, box_size_threshold=None, scaling_factor=None, gpu=0):
        """Setup the object detetor.
        This methods initializes the object detector. The path argument specifies the location
        of the frozen graph file (*.pb) on the hard disk. The gpu argument specifies the id (int)
        of the GPU on which to run the object detector. The function returns a detector object.
        """
        super().__init__(path, box_size_threshold, scaling_factor)
        self.logger = logging.getLogger(__name__)
        # Load frozen tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                with tf.device('/gpu:{}'.format(gpu)):
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            # Get tensors from graph
            tfconfig = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False)
            tfconfig.gpu_options.allow_growth=True
            self.sess = tf.Session(config=tfconfig)
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            self.logger.info(("DetectorTF (ID {}): Initialized object detector "
                "(model path: {}, box size threshold: {}, scaling factor: {}, "
                "gpu: {}).").format(id(self), path, box_size_threshold,
                scaling_factor, gpu))


    def _get_output_dict(self, frame):
        """Get output dictionary of detector for a given frame."""
        # Run inference
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(frame, 0)})
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        n_detections = output_dict['num_detections']
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0][:n_detections]
        output_dict['detection_scores'] = output_dict['detection_scores'][0][:n_detections]
        output_dict['detection_classes'] = output_dict['detection_classes'][0][:n_detections].astype(np.uint8)
        # convert class indices to labels
        cls_labels = []
        for cls_idx in output_dict['detection_classes']:
            cls_labels.append(self._class_index_to_label(cls_idx))
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.str_)
        output_dict['detection_classes'] = cls_labels
        # convert format of bounding boxes to (xmin, ymin, width, height)
        frame_width = np.shape(frame)[1]
        frame_height = np.shape(frame)[0]
        output_dict = self._convert_bounding_box_format(output_dict, frame_width, frame_height)
        return output_dict


    def _convert_bounding_box_format(self, output_dict, frame_width, frame_height):
        """Converts format of all bounding boxes in output_dict.
        Method modifies the bounding box format of all boxes in the detector's output_dict.
        The new format of each box (a row in output_dict['detection_boxes']) is (xmin, ymin, width, heigth).
        Box coordinates are determined based on the size of the frame fed into detect method.
        """
        output_dict['detection_boxes'][:, 0] = np.multiply(output_dict['detection_boxes'][:, 0], frame_height)  # ymin
        output_dict['detection_boxes'][:, 1] = np.multiply(output_dict['detection_boxes'][:, 1], frame_width)  # xmin
        output_dict['detection_boxes'][:, 2] = np.multiply(output_dict['detection_boxes'][:, 2], frame_height)  # ymax
        output_dict['detection_boxes'][:, 3] = np.multiply(output_dict['detection_boxes'][:, 3], frame_width)  # xmax
        new_output_dict = output_dict.copy()
        new_output_dict['detection_boxes'] = np.copy(output_dict['detection_boxes'])
        new_output_dict['detection_boxes'][:, 0] = output_dict['detection_boxes'][:, 1]  # xmin
        new_output_dict['detection_boxes'][:, 1] = output_dict['detection_boxes'][:, 0]  # ymin
        new_output_dict['detection_boxes'][:, 2] = output_dict['detection_boxes'][:, 3] - output_dict['detection_boxes'][:, 1]  # width
        new_output_dict['detection_boxes'][:, 3] = output_dict['detection_boxes'][:, 2] - output_dict['detection_boxes'][:, 0]  # height
        return new_output_dict


#################################################################################
#
#       FASTER R-CNN OBJECT DETECTOR BY GITHUB USER ENDERNEWTON
#
#################################################################################


class DetectorEndernewton(_BaseDetector):
    """Interface for Faster-RCNN object detector by GitHub user Endernewton.
    GitHub: [https://github.com/endernewton/tf-faster-rcnn]
    """

    def __init__(self, path, architecture, anchor_scales=[8, 16, 32], box_size_threshold=None, scaling_factor=None, gpu=0):
        """Setup the object detector.
        This methods initializes the object detector. The path argument specifies the location
        of the detecor's model file (*.ckpt) on the hard disk. The gpu argument specifies the id (int)
        of the GPU on which to run the object detector. The architecture argument specifies
        which model the file represents. Can be "vgg16", "res50" or "mobile". Anchor scales differ
        for different models. For the first gen endernewton models the default can be kept. For the
        second gen models the following settings apply:
        mobile: anchor_scales=[2, 4, 8, 16, 32, 64, 128]
        res50: anchor_scales=[?]
        """
        super().__init__(path, box_size_threshold, scaling_factor)
        self.logger = logging.getLogger(__name__)
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        self.num_classes = 3  # this model can distungiosh between three classes
        if not os.path.isfile(path + '.meta'):
            error_msg = ("{:s} not found.\nDid you download the "
                "proper networks from our server and place them "
                "properly?").format(path + '.meta')
            raise IOError(error_msg)
            self.logger.exception(error_msg)

        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True,
                                  log_device_placement=False)
        tfconfig.gpu_options.allow_growth=True

        # init session
        tf.reset_default_graph()
        self.sess = tf.Session(config=tfconfig)
        # load network
        if architecture == 'vgg16':
            self.net = vgg16()
        elif architecture == 'res50':
            self.net = resnetv1(num_layers=50)
        elif architecture == 'mobile':
            self.net = mobilenetv1()
        else:
            raise NotImplementedError
        with tf.device('/gpu:{}'.format(gpu)):
            self.net.create_architecture("TEST", self.num_classes, tag='default', anchor_scales=anchor_scales)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

        self.logger.info(("DetectorEndernewton (ID {}): Initialized object "
            "detector (model path: {}, architecture: {}, anchor scales: {}, "
            "box size threshold: {}, scaling factor: {}, gpu: {}).").format(
            id(self), path, architecture, anchor_scales, box_size_threshold,
            scaling_factor, gpu))


    def _get_output_dict(self, frame):
        """Get output dictionary of detector for a given frame."""
        scores, boxes = im_detect(self.sess, self.net, frame)
        # filter detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        detections = []
        classes = []
        for cls_ind in range(1, self.num_classes):  # skip brackground class
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            dets = dets[inds]
            detections.append(dets[inds])
            cls_label = self._class_index_to_label(cls_ind)
            classes.append(np.repeat(cls_label, len(dets)))

        # Convert boxes to the output format of tensorflow object detection API
        output_dict = {
            'num_detections': 0,
            'detection_boxes': np.empty(shape=(0, 4), dtype=np.float32),
            'detection_scores': np.empty(shape=(0,), dtype=np.float32),
            'detection_classes': np.empty(shape=(0,), dtype=np.str_)
        }
        detections = np.vstack(detections)
        classes = np.hstack(classes)
        output_dict['num_detections'] = np.shape(detections)[0]
        output_dict['detection_boxes'] = detections[:, :4].astype(np.float32)
        output_dict['detection_scores'] = detections[:, -1].astype(np.float32)
        output_dict['detection_classes'] = classes.astype(np.str_)

        # convert format of bounding boxes to (xmin, ymin, width, height)
        output_dict = self._convert_bounding_box_format(output_dict)

        return output_dict


    def _convert_bounding_box_format(self, output_dict):
        """Converts format of all bounding boxes in output_dict.
        Method modifies the bounding box format of all boxes in the detector's output_dict.
        The new format of each box (a row in output_dict['detection_boxes']) is (xmin, ymin, width, heigth).
        Box coordinates are determined based on the size of the frame fed into detect method.
        """
        output_dict['detection_boxes'][:, 2] = output_dict['detection_boxes'][:, 2] - output_dict['detection_boxes'][:, 0]  # width
        output_dict['detection_boxes'][:, 3] = output_dict['detection_boxes'][:, 3] - output_dict['detection_boxes'][:, 1]  # height
        return output_dict
