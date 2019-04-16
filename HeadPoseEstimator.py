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
xScreen = 640
yScreen = 480


# Camera Calibration

# Intrinsic parameters of camera
center = (xScreen / 2, yScreen / 2)
# Approximate the focal length we do not calibrate the camera beforehand
# Assuming that radial/lens distortion does not exist
# width / depth factor
focal_length = xScreen / np.tan(60 / 2 * np.pi / 180)

# Generate a dummy camera_matrix
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

# Calculate the pitch, yaw, roll of head
# @Params: a video frame, dictionary of detected facical feature points
def find_face_orientation(landmarks):
    # 2D Image points of detected features based on 68 point landmark detection
    # https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/figure_68_markup.jpg
    image_points = np.array([
        (landmarks[30][0], landmarks[30][1]),  # Nose tip
        (landmarks[8][0], landmarks[8][1]),  # Chin
        (landmarks[45][0], landmarks[45][1]),  # Left eye left corner
        (landmarks[36][0], landmarks[36][1]),  # Right eye right corner
        (landmarks[54][0], landmarks[54][1]),  # Left Mouth corner
        (landmarks[48][0], landmarks[48][1])  # Right mouth corner
    ], dtype="double")

    # Generic 3D world coordinates
    # These coordinates were predetermined using a 3D face modeling software
    # Universally accepted for head post estimation.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Make a 4x1 matrix of 0.
    # If dist_coeff is null (set to all 0) solvePnP will assume no lens distortion.
    dist_coeffs = np.zeros((4, 1))

    # Estimate head pose orientation
    # Find rotation and translation vectors.
    (_ , rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Define a 3D axis, for drawing purpose.
    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    # Project 3D points onto 2D image plane
    # Convert world coordinates into camera coordinates
    imgpts, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    #modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # Converts rotation vector into rotation matrix
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    # Concatenate arrays
    proj_matrix = np.hstack((rvec_matrix, translation_vector))

    # Calculate yaw, pitch and roll. rotation_vector from solvePnP are in camera coords
    # Need to convert into real world coords.
    # Refer to
    # http://answers.opencv.org/question/16796/computing-attituderoll-pitch-yaw-from-solvepnp/?answer=52913#post-id-52913
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    # Get angles in world coords
    pitch, yaw, roll = eulerAngles

    return (imgpts, (int(roll), int(pitch), int(yaw)), (landmarks[30][0], landmarks[30][1]))

def print_orientation(rotation, frame):
    roll, pitch, yaw = rotation

    # Centered Head
    if yaw >= -30 and yaw <= 30 and pitch >= 0 and pitch <= 6:
        cv2.putText(frame, 'CENTER', (460, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
    else:
        # Horizontal
        if pitch >= 0 and pitch <= 6:
            if yaw < 0:
                cv2.putText(frame, 'LEFT', (540, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            else:
                cv2.putText(frame, 'RIGHT', (540, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
        # Vertical
        elif yaw >= -30 and yaw <= 30:
            if pitch < 0:
                cv2.putText(frame, 'TOP', (460, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            else:
                cv2.putText(frame, 'BOTTOM', (400, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
        else:
            # Looking in a corner
            # Will print something like 'TOP LEFT'

            if yaw < 0:
                cv2.putText(frame, 'LEFT', (540, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            else:
                cv2.putText(frame, 'RIGHT', (540, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            # Vertical axis
            if pitch < 0:
                cv2.putText(frame, 'TOP', (460, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)
            else:
                cv2.putText(frame, 'BOTTOM', (400, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)

if __name__ == "__main__":
    # Setup Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, xScreen)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, yScreen)

    while(True):
        # Get image captured by webcam
        _, frame = cap.read()

        # Convert screen capture to black-white image, less colors to process = faster results.
        # Also because most opencv functions expect grayscale images.
        # Takes approx 2ms.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect for a face and save the bounding box of the face.
        faces = detector(gray, 0)

        for face in faces:
            # Get face landmarks i.e. right_eye, nose, mouth, etc from face bounding box.
            facial_landmarks = predictor(gray, face)
            # Convert results into an array
            facial_landmarks = face_utils.shape_to_np(facial_landmarks)

            # Attempt to estimate head pose based on detected facial points.
            imgpts, rotation, noseCoords = find_face_orientation(facial_landmarks)

            # Draw orientation vectors stemming from nose point
            # ravel() flattens array, tuple turns array into two parameters
            cv2.line(frame, noseCoords, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
            cv2.line(frame, noseCoords, tuple(imgpts[0].ravel()), (255, 0,), 3)  # BLUE
            cv2.line(frame, noseCoords, tuple(imgpts[2].ravel()), (0, 0, 255), 3)  # RED

            # Display angle infos
            # Diagram of what yall pitch row is:
            # http://1.bp.blogspot.com/-Dew2OIS4T5I/UsX_Fzs2GJI/AAAAAAAAJJI/qZFYrWKjGv8/s1600/ft_kinect.png
            for index in range(len(facial_landmarks)):
                cv2.putText(frame, 'Roll: ' + str(rotation[0]), (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
                cv2.putText(frame, 'Pitch:' + str(rotation[1]), (10, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
                cv2.putText(frame, 'Yaw' + str(rotation[2]), (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

            print_orientation(rotation, frame)


        # show camera feed
        cv2.imshow("Overall", frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()