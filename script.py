import cv2
import numpy as np

def get_threshold(p):
    return 0.2 * int((sum(p)//len(p)))


def video_cap():
    cap = cv2.VideoCapture(0)
    hand_template = cv2.imread("Hand.png",0)
    okay_template = cv2.imread("Okay.png",0)
    peace_template = cv2.imread("Peace.png",0)
    fist_template = cv2.imread("Fist.png",0)
    temp_names = ["open_hand", "okay", "peace", "fist"]
    templates = [hand_template, okay_template, peace_template, fist_template]
    font = cv2.FONT_HERSHEY_SIMPLEX 
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        img = cv2.resize(frame,(1080,720))
        imghsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masked= cv2.inRange(imghsv,(0,70,129),(40,170,200))
        horizontal_p = cv2.reduce(masked, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F).flatten()
        vertical_p = cv2.reduce(masked, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F).flatten()
        avg_h= get_threshold(horizontal_p)
        avg_v= get_threshold(vertical_p)
        top = np.where(horizontal_p>avg_h)[0][0]
        bot = np.where(horizontal_p>avg_h)[0][-1]
        left = np.where(vertical_p>avg_v)[0][0]
        right = np.where(vertical_p>avg_v)[0][-1]
        img = cv2.rectangle(img,(left,top),(right,bot),(0,255,0),2)
        crop = masked[top:bot, left: right]
        cv2.rectangle(img,(left,top), (right,bot), 255, 1)
        c_h,c_w = crop.shape

        results =[]
        for tmp in templates:
            real_temp = cv2.resize(tmp,(c_w//2,c_h//2))
            result = cv2.matchTemplate(crop, real_temp, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            result_t = cv2.matchTemplate(tmp,tmp, cv2.TM_CCOEFF_NORMED)
            results += [max_val]
        name =  temp_names[results.index(max(results))]
        final =cv2.putText(img, name,(30, 30),font,1,(255,0,0),2)
        cv2.imshow("feed", img)

        key = cv2.waitKey(1) & 0xFF
        if key==ord("s"):
            cv2.imwrite("shot.png",img)
        elif key == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # gesture_processing()
    video_cap()
