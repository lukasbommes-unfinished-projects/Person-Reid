import time

import cv2
import numpy as np

from sfmt.config import video_config


class VideoCollector():
    """Interface for collecting video frames."""
    def __init__(self, cams):
        self.cams = cams
        self.caps = []
        # try to open all cameras
        for cap_id, cam in enumerate(self.cams):
            cap = cv2.VideoCapture(cam["source"])
            if cap.isOpened():
                self.caps.append(cap)
            else:
                raise RuntimeError("Could not open camera {}".format(cap_id))


    def get_frame_packet(self):
        """Get frames of all cameras."""
        timestamps = []
        frames = []
        for cap_id, cap in enumerate(self.caps):
            ret_grab = cap.grab()
            timestamp = time.time()
            if ret_grab:
                ret_retrieve, frame = cap.retrieve()
                if ret_retrieve:
                    frames.append(frame)
                    timestamps.append(timestamp)
            if not (ret_grab and ret_retrieve):  # in case of an error
                raise RuntimeError("Could not grab frame of camera {}".format(cap_id))
        return timestamps, frames


    def resize_frames(self, frames, scaling_factor):
        frames_resized = []
        for frame in frames:
            frame_resized = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor)
            frames_resized.append(frame_resized)
        return frames_resized


    def destroy(self):
        """Stops stream sync thread and frees ressources."""
        # clean-up cap objects
        for cap in self.caps:
            cap.release()
