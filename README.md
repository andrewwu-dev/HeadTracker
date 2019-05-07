#**Head Pose Estimator**#
This program determines which part of your computer screen you are looking at.

##**Libraries Used**##
-   [dlib](http://dlib.net/) - Face detection and facial feature prediction

-   opencv - image processing (i.e. convert grayscale to reduce noise, resize images to reduce searching area).
head pose estimation.

##**Notes**##
-   Program calculates pitch, yaw, row of head and check if they are within preset ranges to determine
which section you are looking at.

-   There are 9 sections that can be controlled by head (left,right,up,down,center, corners).

##**Resources**##
- https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

- https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat

- http://answers.opencv.org/question/16796/computing-attituderoll-pitch-yaw-from-solvepnp/?answer=52913#post-id-52913

