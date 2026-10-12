import cv2
import mediapipe as mp
import time
#=================================================================================--
cap = cv2.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands()
ptime = 0
cTime = 0
while True:
    success , img = cap.read()  
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    #print(result.multi_hand_landmarks)
    mpDraw = mp.solutions.drawing_utils

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
           mpDraw.draw_landmarks(img , handLms , mphands.HAND_CONNECTIONS)
           for id , lm in enumerate(handLms.landmark):
               #print(id , lm)
               h , w , c = img.shape
               cx = int(lm.x * w) 
               cy = int(lm.y * h) 
               print(id , cx , cy)
               #if id == 4:
               cv2.circle(img , (cx , cy) , 15 , (255 , 0 , 255) , cv2.FILLED )

#======================================FPS=========================================--
    cTime = time.time()
    fps = 1 / (cTime - ptime)
    ptime = cTime 
    cv2.putText(img , str(int(fps)) , (70 , 70), cv2.FONT_HERSHEY_PLAIN , 1 , (50 , 200 , 100) , 2)
#=================================================================================--
           

            

#=========================information=======location===============================================--




#=======================================fream=========================================--
    cv2.imshow('image' , img)
    #cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('`'):
        break





