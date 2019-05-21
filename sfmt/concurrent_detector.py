import logging
import queue
import threading
from numba import cuda

from sfmt.config import video_config, detector_config
from sfmt.detectors import DetectorTF, DetectorEndernewton


def _make_detector(type, gpu):
    """Initialize and return a new object detector instance."""
    if type == "TF":
        detector = DetectorTF(detector_config["path"],
            box_size_threshold=detector_config["box_size_threshold"],
            scaling_factor=video_config["scaling_factor"],
            gpu=gpu)
    elif type == "EnderNewton":
        detector = DetectorEndernewton(detector_config["path"],
            architecture=detector_config["architecture"],
            anchor_scales=detector_config["anchor_scales"],
            box_size_threshold=detector_config["box_size_threshold"],
            scaling_factor=video_config["scaling_factor"],
            gpu=gpu)
    return detector


class _DetectionThread(threading.Thread):
    """Background thread which runs one detector on a dedicated GPU."""
    def __init__(self, input_queue, output_queue, gpu):
        threading.Thread.__init__(self)
        self.detector = _make_detector(detector_config["type"], gpu)
        self.input_queue = input_queue
        self.output_queue = output_queue


    def run(self):
        while True:
            cap_id, frame = self.input_queue.get()  # do not block upon temrination
            output_dict = self.detector.detect(frame)
            self.output_queue.put((cap_id, output_dict))
            self.input_queue.task_done()


class ConcurrentDetector():
    """Multi-GPU wrapper for object detectors.
    Initializes multiple background threads each of which runs one object
    detector instance on one GPU. The detector type is configured in
    `config.py`. If only one GPU is used, no background threads are created.

    Args:
        num_gpus (`int` or `None`): Number of GPUs on which to run the object
            detector concurrently. Has to be in the range of [1, G] where G
            stands for the number of GPUs available on the machine. If None
            all available GPUs are used and in case of only one available GPU
            an automatic fallback to a non-parallel implementation occurs.

    Raises:
        AssertionError: If `num_gpu` is not an integer and not larger than 0.
        RuntimeError: If no GPUs are available on the system.
    """
    def __init__(self, num_gpus=detector_config["use_num_gpus"]):
        assert (num_gpus is None) or (isinstance(num_gpus, int) and num_gpus > 0), "num_gpus must be a positive integer."
        num_gpus_available = len(cuda.gpus)
        if num_gpus_available < 1:
            raise RuntimeError("There are no GPUs available on your system.")
        if num_gpus is None:
            self.num_gpus = num_gpus_available
        else:
            if num_gpus <= num_gpus_available:
                self.num_gpus = num_gpus
            else:
                raise ValueError(("Currently {} gpus are available for computation. "
                                  "Set the num_gpus argument to a value smaller or "
                                  "equal to this.").format(num_gpus_available))
        self.logger = logging.getLogger(__name__)
        # single GPU fallback
        if self.num_gpus == 1:
            self.detector = _make_detector(detector_config["type"], gpu=0)
            self.logger.info("ConcurrentDetector (ID {}): Intialized. Falling back to single GPU.".format(id(self)))
        # multi GPU
        else:
            self.logger.info("ConcurrentDetector (ID {}): Intialized concurrent detector using {} GPUs.".format(id(self), self.num_gpus))
            self.detector_input_queue = queue.Queue()
            self.detector_output_queue = queue.PriorityQueue()
            for gpu in range(self.num_gpus):
                detection_thread = _DetectionThread(self.detector_input_queue, self.detector_output_queue, gpu)
                detection_thread.daemon = True
                detection_thread.start()
                self.logger.info("ConcurrentDetector (ID {}): Started detection thread with thread-ID {}.".format(id(self), detection_thread.ident))


    def detect_batch(self, frames):
        """Run object detection concurrently on a batch of frames.
        Feeds a list of frames (np.arrays of shape HxWxC) in num_gpu object detectors
        and returns a result list of output_dicts as specified in the detectors module.
        If num_gpu > 1, detection happens in a concurrent fashion. Each entry belongs
        to one of the input frames. They are sorted so that the ith entry in the returned
        list corresponds to the ith frame in the input frame list.

        Args:
            frames (`list` of `numpy.ndarray`): A frame packet as produced by
                the `video_collector`. Each list entry is a frame of shape
                `(H x W x C)` or None in case the frame is invalid.

        Returns:
            last_output_dicts (`list` of `dict`): This list contains a
                dictionary of detection results for each input frame. Each
                dictionary has the following items:
                - 'num_detections' (`int`): Number N of detections per frame.
                - 'detection_boxes' (`numpy.ndarray`): Shape `(N, 2)`. Each
                      row contains bounding box coordinates for one of the
                      detected objects.
                - 'detection_classes' (`list` of `int`): Class of each
                      detected object. Order corresponds to rows of `detection
                      boxes` array.
                - 'detection_scores' (`list` of `float`): Probabilities for
                      each box to contain an object of the predicted class.
        """
        last_output_dicts = [None for _ in range(len(frames))]
        # single GPU fallback
        if self.num_gpus == 1:
            for cap_id, frame in enumerate(frames):
                output_dict = self.detector.detect(frame)
                last_output_dicts[cap_id] = output_dict
        # multi GPU
        else:
            for cap_id, frame in enumerate(frames):
                self.detector_input_queue.put((cap_id, frame))
            # wait until processing is done, then get detection results
            self.detector_input_queue.join()
            while not self.detector_output_queue.empty():
                try:
                    cap_id, output_dict = self.detector_output_queue.get_nowait()
                    last_output_dicts[cap_id] = output_dict
                    self.detector_output_queue.task_done()
                except queue.Empty:
                    continue
        return last_output_dicts
