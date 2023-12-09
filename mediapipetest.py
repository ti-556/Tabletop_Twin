import math
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils #landmarking tool
mp_drawing_styles = mp.solutions.drawing_styles #landmarking style
mp_pose = mp.solutions.pose #pose detection model
 
def getAngle(A, B, C):
    AB = [B[0] - A[0], B[1] - A[1], B[2] - A[2]]
    BC = [C[0] - B[0], C[1] - B[1], C[2] - B[2]]

    # Dot product
    dot_product = AB[0]*BC[0] + AB[1]*BC[1] + AB[2]*BC[2]

    # Magnitudes
    magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2 + AB[2]**2)
    magnitude_BC = math.sqrt(BC[0]**2 + BC[1]**2 + BC[2]**2)

    # Cosine of angle
    cos_angle = dot_product / (magnitude_AB * magnitude_BC)

    # Angle in radians and then in degrees
    angle_radians = math.acos(cos_angle)
    angle_degrees = angle_radians * (180 / math.pi)

    return angle_degrees
 
cap = cv2.VideoCapture(0) #initializing video capture object from webcam
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose: #initializing pose model
  while cap.isOpened(): #while loop until terminated with "esc"
    success, image = cap.read() #retrieve frame
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #coversion from BGR to RGB for pose detection
    results = pose.process(image) #putting frme into pose detection model
    
    #getting landmark attributes:
    leftshoulder = (results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y, results.pose_landmarks.landmark[11].z)
    leftelbow = (results.pose_landmarks.landmark[13].x, results.pose_landmarks.landmark[13].y, results.pose_landmarks.landmark[13].z)
    leftwrist = (results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y, results.pose_landmarks.landmark[15].z)
    leftangle = getAngle(leftshoulder, leftelbow, leftwrist)
    
    rightshoulder = (results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y, results.pose_landmarks.landmark[12].z)
    rightelbow = (results.pose_landmarks.landmark[14].x, results.pose_landmarks.landmark[14].y, results.pose_landmarks.landmark[14].z)
    rightwrist = (results.pose_landmarks.landmark[16].x, results.pose_landmarks.landmark[16].y, results.pose_landmarks.landmark[16].z)
    rightangle = getAngle(rightshoulder, rightelbow, rightwrist)
    
    print(f"left elbow angle: {leftangle}")
    print(f"right elbow angle: {rightangle}")
 
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #conversoin back to BGR
    mp_drawing.draw_landmarks(
        image, #frame
        results.pose_landmarks, #landmarks
        mp_pose.POSE_CONNECTIONS, #skeleton links
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) #default landmark drawing
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1)) #frame flipped and displayed
    if cv2.waitKey(5) & 0xFF == 27: #terminate if "esc" pressed
      break
cap.release()