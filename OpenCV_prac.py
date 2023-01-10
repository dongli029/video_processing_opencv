import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("./original video.mp4")
cap2 = cv2.VideoCapture("./original video.mp4") # 用來分屏
frame_num = 0 # 記錄幀數
print("影片總禎數=", cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 印出影片總禎數  = print("原影片總禎數=", cap.get(7))
sift = cv2.xfeatures2d.SIFT_create() # SIFT 特徵擷取, 抓取邊緣特徵
kernel = np.ones((3, 3), np.uint8)  # SOBEL [3, 3]
i = 1 # 計算旋轉角度

# fps = cap.get(cv2.CAP_PROP_FPS)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('.26_曾東立_HW_output.mp4', fourcc, fps, (480, 480))

while cap.isOpened():
    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (600, 450))
    Canny_img = cv2.Canny(frame1, 100, 200)
    cv2.putText(Canny_img, 'Canny', (500, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(Canny_img, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms',
                (0, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('After process', Canny_img)
    # out.write(Canny_img)
    if cv2.waitKey(25) == 27:
        break
    if ret is None:
        break
    if int(cap.get(1)) > 150:    # 影片0~150禎使用
        break

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (600, 450))
    # 邊緣檢測
    # sobelx = cv2.Sobel(frame, -1, 1, 0, ksize=-1)   # o, ddepth=-1(代表與原影像同深度), dx=1, dy=0, ksize = -1(default)  [Sobel]
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0)   # ddepth = cv2.CV64V
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1)   # ddepth = cv2.CV64V

    sobelx = cv2.convertScaleAbs(sobelx)        # 絕對值, 轉換為cv2.CV_8U
    sobely = cv2.convertScaleAbs(sobely)        # 絕對值, 轉換為cv2.CV_8U
    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0) # 將x,y方向分別抓到的特徵疊圖
    # sobelxy = cv2.bitwise_and(sobelx, sobely)
    cv2.putText(sobelxy, 'Sobel', (500, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(sobelxy, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms',
                (0, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("After process", sobelxy)
    # out.write(sobelxy)
    if cv2.waitKey(25) == 27:
        break
    if ret is None:
        break
    if int(cap.get(1)) > 300:    # 影片150~300禎使用
        break


while cap.isOpened(): # contours detect
    (ret, frame) = cap.read()
    frame = cv2.resize(frame, (600, 450))
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.putText(frame, 'findContours',(400, 440),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255),3)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_1 = cv2.drawContours(frame, cnts, -1, (0, 255, 0), 2)
    cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms',
                (0, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('After process', img_1)
    # out.write(img_1)
    if cv2.waitKey(25) == 27:
        break
    if int(cap.get(1)) > 500: # 影片300~500禎使用
        break
    if not ret:
        break

while cap.isOpened(): # SIFT 特徵擷取
    (ret, frame) = cap.read()
    frame = cv2.resize(frame, (600, 450))
    cv2.putText(frame, 'sift', (500, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms',
                (0, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    frame_num+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('After process', img)
    # out.write(img)
    if cv2.waitKey(25) == 27:
        break
    if int(cap.get(1)) > 600:   # 設定影片450~600禎段落使用此特效
        break
    if not ret:
        break

while cap2.isOpened():
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    # frame = cv2.resize(frame, (480, 480))
    # frame2 = cv2.flip(cv2.resize(frame2, (480, 480)), 1)

    if not (ret & ret2):
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    # frame[:, :, 2] = frame[:, :, 2] - 3
    frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_HSV2BGR), (480,480))
    frame2 = cv2.resize(cv2.cvtColor(frame2, cv2.COLOR_HSV2BGR),(480,480))

    img_video = frame[:, 0:int(frame.shape[1] / 2)]
    img_video2 = frame2[:, int(frame2.shape[1] / 2):]
    img_out = np.hstack((img_video, img_video2))
    cv2.putText(img_out, 'Split Screen', (280, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img_out, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms',
                (0, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
    cv2.imshow("After process", img_out)
    # out.write(img_out)
    if cv2.waitKey(25) == 27:
        break
    i += 3

    if int(cap.get(1)) > 800:  # 設定影片禎段落600~800使用此特效
        break
    if not ret:
        break

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (600, 450))
    lap = cv2.Laplacian(frame, cv2.CV_64F)
    cv2.putText(lap, 'lap', (500, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
    cv2.putText(lap, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms',
                (0, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('After process', lap)
    # out.write(lap)
    if cv2.waitKey(25) == 27:
        break
    if ret is None:
        break
    if int(cap.get(1)) > 1000:    # 影片800~1000禎使用
        break

while cap.isOpened():  # rotate frame
    (ret, frame) = cap.read()
    frame = cv2.resize(frame, (600, 450))
    cv2.putText(frame, 'warpAffine', (400, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
    cv2.putText(frame, f'{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} frames, {cap.get(cv2.CAP_PROP_POS_MSEC):.0f} ms',
                (0, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    M = cv2.getRotationMatrix2D((300, 225), i, 1)    # rotation
    output = cv2.warpAffine(frame, M, (600, 450))
    cv2.imshow('After process', output)
    # out.write(output)
    if cv2.waitKey(25) == 27:
        break
    i+=3

    if int(cap.get(1)) > 1500:
        break
    if not ret:
        break

cap.release()
cap2.release()
# out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)