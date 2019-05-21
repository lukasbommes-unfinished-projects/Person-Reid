import glob
import pickle
import cv2
import numpy as np

from sfmt.config import video_config, detector_config
from sfmt.concurrent_detector import ConcurrentDetector
from sfmt.video_collector import VideoCollector


if __name__ == "__main__":

    # find detection files
    det_files = glob.glob("detections/*.pkl")

    # load file contents
    detections = []
    for det_file in det_files:
        detection = pickle.load(open(det_file, "rb"))
        detections.append(detection)

    # for loading and showing video frame
    cams = video_config["cams"]
    scaling_factor = video_config["scaling_factor"]
    video_collector = VideoCollector(cams)
    for cap_id in range(len(cams)):
        cv2.namedWindow("cap {}".format(cap_id), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cap {}".format(cap_id), 640, 360)

    # loop over the detections
    for step, detection in enumerate(detections):

        # get frames from each video file
        timestamps, frames = video_collector.get_frame_packet()
        frames_small = video_collector.resize_frames(frames, video_config["scaling_factor"])

        print(step)
        print(timestamps)
        print(detection)


         # at this point the following variables can be used
         # step: current time step (integer)
         # timestamps: list of timestamps of each video frame
         # frames: list of frames of each camera at the time step
         # frames_small: smaller version of the frames
         # detection: a list of detection results (dictionaries) for the given time step


        # display frames
        for cap_id, (frame_small, cam) in enumerate(zip(frames_small, video_config["cams"])):
            cv2.imshow("cap {}".format(cap_id), frame_small)

        # if user presses key 'q' exit the loop and stop program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
