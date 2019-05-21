## Quickstart


### Build the container (only once)

`cd` into project root directory and run

```
sudo docker-compose build
```
This takes a while.


### Start the container

`cd` into project root directory and run

```
xhost +
```
followed by
```
sudo docker-compose up
```


### Open an interactive bash session within running container

Open a new terminal in the project root and type
```
sudo docker exec -it person_reid bash
```
This opens a bash prompt within the container.


## Setting input videos
To use a new set of input videos copy the video files into the "videos" folder
and update the option `video_config["cams"]` in the file `sfmt/config`
accordingly.


## Extracting object detections

To extract objects from a pair of videos in the "videos" folder run the command
```
python3 detection.py
```
in the previously opened interactive bash session within the container.
For each time step of the videos a pickled file containing the detected objects
will be created in the "detections" folder. The naming of the files follows the
schema "step_x" indicating the time step of the contained detections.


## Using the extracted detections

The `reid.py` contains a demo of how to use the extracted detections. Run it via
```
python3 reid.py
```
in the interactive bash session in the container.
