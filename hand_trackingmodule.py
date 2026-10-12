import cv2
import mediapipe as mp
import time
#=================================================================================--


class handdetector():
    def __init__(self , mode = False , maxhands = 2 , detectioncon = 0.5 , trackcon = 0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mphands = mp.solutions.hands
        #self.hands = self.mphands.Hands(self.mode , self.maxhands , self.detectioncon , self.trackcon )
        self.hands = self.mphands.Hands(
    static_image_mode=self.mode,
    max_num_hands=self.maxhands,
    min_detection_confidence=self.detectioncon,
    min_tracking_confidence=self.trackcon
)
#ptime = 0
#cTime = 0

    def findehands(self , img , draw = True):

            imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
            self.result = self.hands.process(imgRGB)
            #print(result.multi_hand_landmarks)
            self.mpDraw = mp.solutions.drawing_utils

            if self.result.multi_hand_landmarks:
                for handLms in self.result.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img , handLms , self.mphands.HAND_CONNECTIONS)
            return img

    def findeposition(self , img , draw=True , handno=0):
        lmlist = []
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handno]
            for id , lm in enumerate(myhand.landmark):           
                                                 
                #print(id , lm) 
                h , w , c = img.shape
                cx = int(lm.x * w) 
                cy = int(lm.y * h) 
                print(id , cx , cy)
                lmlist.append([id , cx , cy])
                #if id == 4:
                if draw:
                    cv2.circle(img , (cx , cy) , 15 , (255 , 0 , 255) , cv2.FILLED )

        return lmlist
                   
              
          
        
#=======================================module=========================================--
def main():

    ptime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handdetector()
    while True:

        success , img = cap.read() 
        img = detector.findehands(img)
        lmlist = detector.findeposition(img)

        if len(lmlist) != 0 :
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - ptime)
        ptime = cTime 
        cv2.putText(img , str(int(fps)) , (70 , 70), cv2.FONT_HERSHEY_PLAIN , 1 , (50 , 200 , 100) , 2)


        cv2.imshow('image' , img)
        if cv2.waitKey(1) & 0xFF == ord('`'):
            break

if __name__ == "__main__":
    main()