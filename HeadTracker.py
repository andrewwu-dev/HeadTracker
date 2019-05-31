# Imports:
# Secondary image processing library
from imutils import face_utils
import math
import numpy as np
# Facial feature detection. dlib contains a bunch of machine learning algorithms.
import dlib
# Image processing
import cv2

# Setup Model
# This a pretrained model obtained from
# https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
landmark = "shape_predictor_68_face_landmarks.dat"
# Uses Histogram of Gradients (HOG) method to detect faces/head
detector = dlib.get_frontal_face_detector()
# Pass in 68 landmark model set so predictor know how to
# recognize facial features.
predictor = dlib.shape_predictor(landmark)

# Screen Size
# Resize to smaller resolution in order to increase speed
xScreen = 640
yScreen = 480

# Camera Calibration

# Intrinsic parameters of camera
center = (xScreen / 2, yScreen / 2)
# Approximate the focal length we do not calibrate the camera beforehand
focalLength = xScreen

# Generate a dummy camera_matrix
cameraMatrix = np.array(
    [[focalLength, 0, center[0]],
     [0, focalLength, center[1]],
     [0, 0, 1]], dtype="double"
)

# Calculate the pitch, yaw, roll of head
def find_face_orientation(landmarks):
    # 2D Image points of detected features based on 68 point landmark detection
    detectedPoints = np.array([
        (landmarks[30][0], landmarks[30][1]),  # Nose tip
        (landmarks[8][0], landmarks[8][1]),  # Chin
        (landmarks[45][0], landmarks[45][1]),  # Left eye left corner
        (landmarks[36][0], landmarks[36][1]),  # Right eye right corner
        (landmarks[54][0], landmarks[54][1]),  # Left Mouth corner
        (landmarks[48][0], landmarks[48][1])  # Right mouth corner
    ], dtype="double")

    # Generic 3D world coordinates
    modelPoints = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Make a 4x1 matrix of 0.
    # If dist_coeff is null (set to all 0) solvePnP will assume no lens distortion.
    distCoeffs = np.zeros((4, 1))

    # Estimate head pose orientation
    # Find rotation and translation vectors.
    (_ , rotationVect, translationVect) = cv2.solvePnP(modelPoints, detectedPoints, cameraMatrix,
                                                                  distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Define a 3D axis
    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    # Project 3D points onto 2D image plane
    # Convert world coordinates into camera coordinates
    imgpts, _ = cv2.projectPoints(axis, rotationVect, translationVect, cameraMatrix, distCoeffs)
  
    # Converts rotation vector into rotation matrix
    rvecMat = cv2.Rodrigues(rotationVect)[0]
    # Concatenate arrays
    projMat = np.hstack((rvecMat, translationVect))

    # Calculate yaw, pitch and roll. rotation_vector from solvePnP are in camera coords
    # Need to convert into real world coords.
    # Refer to
    # http://answers.opencv.org/question/16796/computing-attituderoll-pitch-yaw-from-solvepnp/?answer=52913#post-id-52913
    eulerAngles = cv2.decomposeProjectionMatrix(projMat)[6]

    # Get angles in world coords
    pitch, yaw, roll = eulerAngles

    return (imgpts, (int(roll), int(pitch), int(yaw)), (landmarks[30][0], landmarks[30][1]))

def print_orientation(rotation, frame):
    roll, pitch, yaw = rotation

    pitch = pitch * -1;

    upperBound = 3;
    lowerBound = -5;
    leftBound = -30;
    rightBound = 25;

    # Centered Head
    if yaw >= leftBound and yaw <= rightBound and pitch <= upperBound and pitch >= lowerBound:
        cv2.putText(frame, 'CENTER', (460, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
    else:
        # Centered Horizontally
        if pitch <= upperBound and pitch >= lowerBound:
            if yaw < leftBound:
                cv2.putText(frame, 'LEFT', (540, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            elif yaw > rightBound:
                cv2.putText(frame, 'RIGHT', (540, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
        # Centered Vertically
        elif yaw >= leftBound and yaw <= rightBound:
            if pitch > upperBound:
                cv2.putText(frame, 'TOP', (460, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            elif pitch < lowerBound:
                cv2.putText(frame, 'BOTTOM', (400, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
        else:
            # Looking in a corner
            # Will print something like 'TOP LEFT'
           
            # Vertical axis
            if pitch > upperBound:
                cv2.putText(frame, 'TOP', (460, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            elif pitch < lowerBound:
                cv2.putText(frame, 'BOTTOM', (400, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            # Horizontal Axis
            if yaw < leftBound:
                cv2.putText(frame, 'LEFT', (540, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            elif yaw > rightBound:
                cv2.putText(frame, 'RIGHT', (540, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            

if __name__ == "__main__":
    # Setup Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, xScreen)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, yScreen)

    while(True):
        # Get image captured by webcam
        _, frame = cap.read()

        # Reduce noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect for a face and save the bounding box of the face.
        faces = detector(gray, 0)

        for face in faces:
            # Get face landmarks i.e. right_eye, nose, mouth, etc from face bounding box.
            facialLandmarks = predictor(gray, face)
            # Convert results into an array
            facialLandmarks = face_utils.shape_to_np(facialLandmarks)

            # Attempt to estimate head pose based on detected facial points.
            imgpts, rotation, noseCoords = find_face_orientation(facialLandmarks)

            # Draw orientation vectors stemming from nose point
            # ravel() flattens array, tuple turns array into two parameters
            cv2.line(frame, noseCoords, tuple(imgpts[2].ravel()), (255, 0, 0), 3)

            # Display angle infos
            # Diagram of what yall pitch row is:
            # http://1.bp.blogspot.com/-Dew2OIS4T5I/UsX_Fzs2GJI/AAAAAAAAJJI/qZFYrWKjGv8/s1600/ft_kinect.png
            cv2.putText(frame, 'ROLL: ' + str(rotation[0]), (460, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            cv2.putText(frame, 'PITCH:' + str(rotation[1]), (460, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            cv2.putText(frame, 'YAW' + str(rotation[2]), (460, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)

            print_orientation(rotation, frame)


        # show camera feed
        cv2.imshow("Frame", frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
