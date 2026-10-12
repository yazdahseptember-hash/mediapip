import cv2
import mediapipe as mp
import time
#-----------------------------------------------------------
cap = cv2.VideoCapture('posevideo/1.mp4')

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()



cTime = 0
pTime = 0
while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img , result.pose_landmarks , mpPose.POSE_CONNECTIONS)
        print(result.pose_landmarks)

    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img , str(int(fps)) , (70 , 50) , cv2.FONT_HERSHEY_PLAIN, 3 , (255 , 0 , 0 ) , 3)

    cv2.imshow("image" , img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('`'):
        break

