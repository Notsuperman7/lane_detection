# Lane Detection Node

This package contains a ROS 2 Jazzy node used for an Autonomous Mobile Robot (AMR).  
The node detects red lane tape from a camera feed and publishes lane position information for robot navigation.

## Overview

The `lane_detection_node` subscribes to the robot camera topic, processes the image using OpenCV, detects the lane, and calculates the robot’s error from the lane center.

It is designed to help an autonomous mobile robot follow a lane using visual feedback.

## Technologies Used

- ROS 2 Jazzy
- Python 3
- OpenCV
- NumPy
- cv_bridge

## ROS 2 Node

### Node Name

```bash
lane_detection_node

Subscribed Topic
/camera/image_raw    // the camera topic, adjust it to ur camera topic

this node Publish in topic " /lane_detection/lane_info ":
Message type:
geometry_msgs/msg/Vector3  :
The published values are:

x: normalized lane error
y: confidence value
z: lane detection status
1.0 = lane detected
0.0 = lane not detected


"/lane_detection/debug_image"
this topic i used it to publish the processed images to my webGUI so i can get a Live Detection for the lane

How It Works
1-Receives camera frames from /camera/image_raw.
2-Crops the lower part of the image as the region of interest.
3-Converts the image from BGR to HSV.
4-Applies a red color mask to detect red tape.
5-Uses morphology operations to clean the mask.
6-Finds lane contours.
7-Estimates the lane center.
6-Publishes the normalized error, confidence, and detection status.


Debug Output

The node also saves debug frames, ROI images, and masks inside:
/home/pi5/autonomouse_ws/src/camera_pkg/live_debug    // agjust it to ur file location
the Purpose of this is for debuging the processed images output to tune the mask if the mask wasn't good enough

"YOU CAN ADJUST THE MASK TO ANY LANE COLOUR YOU WANT AND IT WOULD WORK WITH YOU!!! 😉😉"

