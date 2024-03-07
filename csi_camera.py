import sys
import cv2

def gstreamer_pipeline(
        sensor_id=0,
        capture_width=1280,
        capture_height=720,
        display_width=640,
        display_height=480,
        framerate=60,
        flip_method=0,
):
    return (
            "nvarguscamerasrc sensor-id=%d !"
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
            "appsink max-buffers=1 drop=True "
            % (
                sensor_id,
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )

def main():
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        try:
            while True:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow("video", frame)  # 顯示視頻幀
                else:
                    print("Not read")

                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Bye")
        finally:
            cap.release()
            cv2.destroyAllWindows()  # 關閉窗口
    else:
        print("Unable to open camera")

if __name__ == "__main__":
    main()
