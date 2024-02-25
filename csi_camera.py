import cv2


# 設定gstreamer管道參數
def gstreamer_pipeline(
        capture_width=1280,  # 相機預先擷取的影像寬度
        capture_height=720,  # 相機預先擷取的影像高度
        display_width=1280,  # 視窗顯示的影像寬度
        display_height=720,  # 視窗顯示的影像高度
        framerate=60,  # 捕捉幀率
        flip_method=0,  # 是否旋轉影像
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


if __name__ == "__main__":
    capture_width = 1280
    capture_height = 720
    display_width = 1280
    display_height = 720
    framerate = 60
    flip_method = 0

    # 创建管道
    print(gstreamer_pipeline(capture_width, capture_height, display_width, display_height, framerate, flip_method))

    # 管道與視訊串流綁定
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)

        # 逐幀顯示
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)

            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:  # ESC鍵退出
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("打開相機失敗")
