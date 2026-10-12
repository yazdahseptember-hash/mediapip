import cv2
import mediapipe as mp
import time


class handdetector():

    def __init__(self, mode=False, maxhands=2, detectioncon=0.5, trackcon=0.5):

        self.mode = mode
        self.maxhands = maxhands
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mphands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils

        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxhands,
            min_detection_confidence=self.detectioncon,
            min_tracking_confidence=self.trackcon
        )

        self.result = None


    # تشخیص دست
    def find_hands(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.result = self.hands.process(imgRGB)

        return img


    # رسم خطوط و نقاط دست
    def draw_hands(self, img):

        if self.result.multi_hand_landmarks:

            for handLms in self.result.multi_hand_landmarks:

                self.mpDraw.draw_landmarks(
                    img,
                    handLms,
                    self.mphands.HAND_CONNECTIONS
                )

        return img


    # گرفتن مختصات نقاط دست
    def find_position(self, img, handno=0):

        lmlist = []

        if self.result.multi_hand_landmarks:

            myhand = self.result.multi_hand_landmarks[handno]

            for id, lm in enumerate(myhand.landmark):

                h, w, c = img.shape

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                lmlist.append([id, cx, cy])

        return lmlist


    # رسم دایره روی نقاط دست
    def draw_points(self, img, lmlist):

        for id, cx, cy in lmlist:

            cv2.circle(
                img,
                (cx, cy),
                15,
                (255, 0, 255),
                cv2.FILLED
            )

        return img


# ======================================= MAIN =======================================

def main():

    ptime = 0

    cap = cv2.VideoCapture(0)

    detector = handdetector()

    while True:

        success, img = cap.read()

        # تشخیص دست
        detector.find_hands(img)

        # رسم دست
        detector.draw_hands(img)

        # گرفتن مختصات نقاط
        lmlist = detector.find_position(img)

        # رسم نقاط
        detector.draw_points(img, lmlist)

        if len(lmlist) != 0:
            print(lmlist[4]) 

        # ========================== FPS ==========================

        cTime = time.time()

        fps = 1 / (cTime - ptime)

        ptime = cTime

        cv2.putText(
            img,
            str(int(fps)),
            (70, 70),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (50, 200, 100),
            2
        )

        # ========================== FRAME ==========================

        cv2.imshow("image", img)

        if cv2.waitKey(1) & 0xFF == ord('`'):
            break


if __name__ == "__main__":





    main()


