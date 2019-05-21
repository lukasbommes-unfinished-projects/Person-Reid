import time
import pickle

import cv2
import numpy as np

from sfmt.config import video_config, detector_config
from sfmt.concurrent_detector import ConcurrentDetector
from sfmt.video_collector import VideoCollector


if __name__ == "__main__":

    cams = video_config["cams"]
    scaling_factor = video_config["scaling_factor"]

    # for retrieval of video frames from files/cameras
    video_collector = VideoCollector(cams)

    # create object detector
    detector = ConcurrentDetector()

    # create GUI window for video output
    for cap_id in range(len(cams)):
        cv2.namedWindow("cap {}".format(cap_id), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cap {}".format(cap_id), 640, 360)

    step = 0
    while True:
        # get frames from each camera (each list entry corresponds to a frame of a different camera)
        timestamps, frames = video_collector.get_frame_packet()

        # resize the frame by a factor to make subsequent processing faster
        frames_small = video_collector.resize_frames(frames, video_config["scaling_factor"])

        # run the object detector on each frame in the list of frames
        # returns a set of detected bounding boxes, class labels and confidence scores for each frame
        detections = detector.detect_batch(frames)

        # store detections for each step in a file "detections/step_xxx"
        print(detections)
        pickle.dump(detections, open("detections/step_{}.pkl".format(step), "wb"))

        step += 1

        # display frames
        for cap_id, (frame_small, cam) in enumerate(zip(frames_small, video_config["cams"])):
            cv2.imshow("cap {}".format(cap_id), frame_small)

        # if user presses key 'q' exit the loop and stop program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # clean up
    video_collector.destroy()
    cv2.destroyAllWindows()
