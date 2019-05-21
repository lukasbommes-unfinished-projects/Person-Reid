#################################################################################
#
#       VIDEO CONFIGURATION
#
#################################################################################

video_config = {}

# Camera configuration
video_config["cams"] = [
    # cam_0
    {"source": "videos/output_C3_part_5.avi",
     "calibration_parameters": "camera_calibration/calibration_parameters/0",
     "frame_width": 1920,
     "frame_height": 1080,
     "frame_rate": 15.0},
    # cam 1
    {"source": "videos/output_C7_part_5.avi",
     "calibration_parameters": "camera_calibration/calibration_parameters/1",
     "frame_width": 1920,
     "frame_height": 1080,
     "frame_rate": 15.0}
]

# Scaling factor [0,1] to rescale the input video frames prior to tracking (DEFAULT: 0.4)
video_config["scaling_factor"] = 0.3


#################################################################################
#
#       DETECTOR CONFIGURATION
#
#################################################################################

detector_config = {}

# Which detector type to use. Can be "TF" / "EnderNewton" / "YOLOV3"
detector_config["type"] = "EnderNewton"

# Type specific detector options
if detector_config["type"] == "TF":
    detector_config["path"] = "models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb"  # detector frozen inferenze graph (*.pb)
    detector_config["box_size_threshold"] = (0.25*1920, 0.6*1080) # discard detection boxes larger than this threshold
    detector_config["track_classes"] = ["person"]  # track only objects with the following class labels ([] to disable)

elif detector_config["type"] == "EnderNewton":
    detector_config["path"] = "models/endernewton/mobile/mobile_faster_rcnn_iter_70000.ckpt"  # detector model file (*.ckpt)
    detector_config["architecture"] = "mobile"  # Backbone architecture. Either "mobile" or "res50"
    detector_config["anchor_scales"] = [2, 4, 8, 16, 32, 64, 128]  # change for res50
    detector_config["box_size_threshold"] = None
    detector_config["track_classes"] = []

elif detector_config["type"] == "YOLOV3":
    detector_config["path"] = "models/yolov3/yolov3-sfmt_final.weights"  # detector model file (*.weights)
    detector_config["cfg_file"] = "models/yolov3/yolov3-sfmt.cfg"  # detector configuration file (*.cfg)
    detector_config["obj_file"] = "models/yolov3/sfmt.data"  # detector objects file (*.data)
    detector_config["helmet_file"] = "models/yolov3/helmet.h5"  # detector objects file (*.data)
    detector_config["clothes_file"] = "models/yolov3/clothes.h5"  # detector objects file (*.data)
    detector_config["box_size_threshold"] = None
    detector_config["track_classes"] = []

# The detector is run on every Nth frame, where N is this number (DEFAULT: 2)
detector_config["detector_cycle"] = 5

# The number of GPUs to use. Must be an integer larger than 0 and smaller or equal to the
# number of available CUDA GPUs. Set to None to use all available GPUs and automatically
# fall back to a non-parallel approach if only one GPU is available.
detector_config["use_num_gpus"] = None
