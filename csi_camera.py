import cv2

# Reading BGRx frames into opencv:
cap = cv2.VideoCapture(
    "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=3264, height=2464, framerate=21/1, format=NV12 ! nvvidconv ! "
    "video/x-raw,format=BGRx ! appsink drop=1",
    cv2.CAP_GSTREAMER)

# Or reading BGR frames as expected by most opencv algorithms: cap = cv2.VideoCapture("nvarguscamerasrc !
# video/x-raw(memory:NVMM),width=3264, height=2464, framerate=21/1, format=NV12 ! nvvidconv ! video/x-raw,format=BGRx
# ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("failed to open video capture")
    exit(-1)

cv2.namedWindow("CamPreview", cv2.WINDOW_AUTOSIZE)

frames = 0
# Run for 10s @21fps
while frames < 210:
    ret_val, img = cap.read()
    if not ret_val:
        print("failed to read from video capture")
        break
    frames = frames + 1

    cv2.imshow('CamPreview', img)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
