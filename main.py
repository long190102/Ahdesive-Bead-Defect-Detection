import cv2 as cv
import numpy as np
import os
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import threading
from serial_communication import *

# Khởi tạo các biến toàn cục
# cap = cv.VideoCapture(1)  # Mở camera USB (camera thứ hai)
cap = cv.VideoCapture("videotest_chuan.avi")  # Mở video từ file


frame = None  # Khung hình hiện tại từ camera
recap_frame = None  # Ảnh recap rõ nét nhất
highest_sharpness = 0  # Độ nét cao nhất của khung hình recap
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 200

# Tạo thư mục lưu ảnh recap nếu chưa tồn tại
RECAP_FOLDER = "RECAP_OPENCV"
os.makedirs(RECAP_FOLDER, exist_ok=True)

# Thiết lập độ phân giải cho camera
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1000 / 1.5)


def calculate_sharpness(image):
    """
    Tính độ nét của một khung hình bằng Laplacian.
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(gray, cv.CV_64F).var()


def detect_object_center(center, up_border, low_border):
    """
    Phát hiện vật thể trong khung hình và kiểm tra xem nó có ở giữa khung hình không.
    """
    if up_border < center[1] < low_border:
        return True  # Vật thể nằm ở giữa khung hình
    return False


def camera_thread():
    """
    Luồng đọc camera và cập nhật khung hình.
    """
    global frame, cap
    while True:
        ret, temp_frame = cap.read()
        if ret:
            frame = temp_frame
            delay_time = 1 / cap.get(cv.CAP_PROP_FPS) * 1000
            cv.waitKey(int(delay_time))
        else:
            # Nếu không nhận được frame, khởi động lại camera
            print("Khởi động lại camera...")
            cap.release()
            time.sleep(1)  # Tạm nghỉ trước khi khởi động lại
            cap = cv.VideoCapture(1)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)


def get_Cropped_image(image):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    _, binary_image = cv.threshold(gray_image, 150, 255, cv.THRESH_BINARY)
    binary_image[:, :200] = 0
    binary_image[:, 500:] = 0

    edges = cv.Canny(binary_image, 90, 200)

    # Create a black mask with the same dimensions as the image
    mask = np.zeros_like(binary_image, dtype=np.uint8)

    width_1 = 90
    height_1 = 145
    width_2 = height_1
    height_2 = 210
    left_border = 200
    right_border = 430

    # Find contours in the edge-detected image
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Get the minimum area bounding rectangle (rotated rectangle)
        rect = cv.minAreaRect(contour)

        # Get the box points of the rectangle
        box_points = cv.boxPoints(rect)
        box_points = np.int0(box_points)  # Convert to integer coordinates

        # Get the center, dimensions, and angle of the bounding box
        (x, y), (w, h), angle = rect
        (width, height) = (w, h) if w < h else (h, w)
        # if (((width>width_1-20 and width <width_1+20) and (height_1>height_1-20 and height<height_1+20)) or (
        #   (width>width_2-20 and width<width_2+20) and (height>height_2-20 and height<height_2+20))) and (
        #     x > left_border and x< right_border
        # ):
        # if(width>0):
        if width > 70:

            # Draw the rotated bounding box on the mask (white-filled polygon)
            cv.fillPoly(mask, [box_points], (255, 255, 255))

    # Apply the mask to keep only the areas inside the rectangles
    result = cv.bitwise_and(binary_image, mask)

    # cv.imshow("result", result)
    # cv.waitKey(0)
    # Create a structuring element (kernel)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))  # 5x5 square

    # Apply dilation
    dilated = cv.dilate(result, kernel, iterations=3)

    # Apply erosion
    eroded = cv.erode(result, kernel, iterations=3)
    final = cv.erode(dilated, kernel, iterations=3)

    filed_binary_image = final.copy()
    # cv.imshow("image", filed_binary_image)
    # cv.waitKey(0)
    edges = cv.Canny(filed_binary_image, 90, 200)
    contour, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Get the minimum area bounding rectangle (rotated rectangle)
    try:
        rect = cv.minAreaRect(contour[0])
    except IndexError:
        rect = (0, 0), (0, 0), 0
    # Get the center, dimensions, and angle of the bounding box
    center, (width, height), angle = rect

    # if(width>width_2-20 and width<width_2+20) and (height>height_2-20 and height<height_2+20):

    cropped = None
    if width > height:
        width, height = height, width
        angle += 90  # Adjust the angle to align the longer side vertically

    # Step 7: Get the rotation matrix to rotate the image
    M = cv.getRotationMatrix2D(center, angle, 1.0)

    # Step 8: Rotate the image to align the longer side vertically
    rotated_image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Step 9: Define the cropping box (after rotating, the bounding box should align with the axis)
    cropped = rotated_image[
        int(center[1] - height // 2) : int(center[1] + height // 2),
        int(center[0] - width // 2) : int(center[0] + width // 2),
    ]

    # Ensure the longer edge is vertical
    # if cropped.shape[1] < cropped.shape[0]:  # If width > height
    #     cropped = cv.rotate(cropped, cv.ROTATE_90_CLOCKWISE)
    # # Add text to the image
    cropped = cv.rotate(cropped, cv.ROTATE_90_CLOCKWISE)
    return cropped, center


def change_outside_contour_to_white(image, contour):
    """
    Changes every pixel outside the given contour to white.

    :param image: Input image (numpy array)
    :param contour: Contour (numpy array) to retain the region inside
    :return: Processed image with outside pixels turned white
    """
    # Create a blank mask with the same dimensions as the image
    mask = np.zeros_like(image, dtype=np.uint8)

    # Fill the contour area in the mask with white
    if len(image.shape) == 3:  # Color image
        mask = cv.drawContours(
            mask, [contour], -1, (255, 255, 255), thickness=cv.FILLED
        )
    else:  # Grayscale image
        mask = cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)

    # Apply the mask to the image
    result = cv.bitwise_and(image, mask)

    # Turn outside contour pixels to white
    white_background = np.ones_like(image, dtype=np.uint8) * 255
    result = np.where(mask > 0, result, white_background)

    return result


def add_border(binary_image, thickness):
    """
    Adds a border of the specified thickness to a binary image.
    The output image will have the same shape as the input, and the border will replace the original pixel values.

    Parameters:
        binary_image (numpy.ndarray): Input binary image (single channel, 0 and 255 values).
        thickness (int): Thickness of the border in pixels.

    Returns:
        numpy.ndarray: Binary image with the border added, having the same shape as the input.
    """
    if len(binary_image.shape) != 2:
        raise ValueError("Input image must be a single-channel binary image.")

    if thickness <= 0:
        raise ValueError("Thickness must be a positive integer.")

    # Create a new image with the same shape as the input
    bordered_image = np.zeros_like(binary_image, dtype=np.uint8)

    # Set the border area to white (255)
    bordered_image[:thickness, :] = 255  # Top border
    bordered_image[-thickness:, :] = 255  # Bottom border
    bordered_image[:, :thickness] = 255  # Left border
    bordered_image[:, -thickness:] = 255  # Right border

    # Copy the original image content to the non-border area
    bordered_image[thickness:-thickness, thickness:-thickness] = binary_image[
        thickness:-thickness, thickness:-thickness
    ]

    return bordered_image


def divide_shape_by_bounding_height(contour, x, image_shape):
    """
    Divide the shape bounded by a contour into `x` equal-height parts.

    :param contour: Input contour (numpy array of shape [n, 1, 2])
    :param x: Number of equal-height parts to divide the shape into
    :param image_shape: Shape of the image (height, width) for creating masks
    :return: List of contours representing boundaries of the divided regions
    """
    # Get the bounding rectangle of the contour
    x_min, y_min, width, height = cv.boundingRect(contour)
    segment_height = height / x

    # Create a mask for the full shape
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)

    divided_contours = []

    # Divide the bounding rectangle into `x` equal-height segments
    for i in range(x):
        # Define the current segment's y-range
        y_start = int(y_min + i * segment_height)
        y_end = int(y_min + (i + 1) * segment_height)

        # Create a mask for the current segment
        segment_mask = np.zeros_like(mask)
        segment_mask[y_start:y_end, x_min : x_min + width] = 255

        # Intersect the segment mask with the shape mask
        intersected_mask = cv.bitwise_and(mask, segment_mask)

        # Find contours in the intersected mask
        segment_contours, _ = cv.findContours(
            intersected_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # Add the largest contour (or all contours if necessary)
        if segment_contours:
            largest_segment = max(segment_contours, key=cv.contourArea)
            divided_contours.append(largest_segment)

    return divided_contours


def get_binary_image(img, threshold=140, thickness=7):

    # img = cv.imread('data_keo/dutdoan/WIN_20241220_10_51_33_Pro.jpg', cv.IMREAD_COLOR)#link anh
    img, _ = get_Cropped_image(img)
    # cv.imshow("image", img)
    # cv.waitKey(0)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray_pre = cv.GaussianBlur(img_gray, (11, 11), 0)
    img_gray_pre = cv.medianBlur(img_gray, 3)
    # cv.imshow('image',img_gray_pre)
    # cv.waitKey(0)
    # Ảnh binary 1: dùng để lấy đường keo (đường keo đen, còn lại trắng)
    thresh, im_bw = cv.threshold(
        img_gray_pre, threshold, 255, cv.THRESH_BINARY
    )  # tất cả các giá trị > 90 thì thành trắng, nhỏ hơn 90 thì đen
    im_bw = add_border(im_bw, thickness)
    w, h = im_bw.shape
    im_bw = cv.resize(im_bw, (0, 0), fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    img = cv.resize(img, (0, 0), fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    return im_bw, img


def detect_error(img, width=27):
    im_bw, img = get_binary_image(img)
    length = im_bw.shape[0]

    # cv.imshow("image_binary", im_bw)
    # cv.waitKey(0)

    edge = cv.Canny(im_bw, 100, 200)
    # cv.imshow("image", im_bw)
    # cv.imshow("image", edge)
    # cv.imwrite('edge.jpg', edge)
    cv.waitKey(0)
    forgive_low = 4
    forgive_high = 6
    num = 0
    contours, hierarchy = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # for contour in contours:
    #     if cv.contourArea(contour) > 1000:
    #          num = num  + 1
    #          cv.drawContours(img, contour, -1, (0, 255, 0), 3)
    # print(num)
    w_error_small = 0
    w_error_big = 0
    l_error_small = 0
    l_error_big = 0
    contournum = 0
    discrete_err = 0
    # for contour in contours:
    # print(cv.contourArea(contour))
    for contour in contours:
        # contournum = 0
        # print(cv.contourArea(contour))
        if cv.contourArea(contour) > 1300:
            contournum = contournum + 1
        # print(contours)
        # print(f'contour num: {contournum}')
        # print(contournum)
        if contournum > 2:
            # if contournum >1:
            # print('hello')
            # print(f"số contour là {len(contours)}")
            cv.putText(
                img, "Dut doan", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
            )
            cv.drawContours(img, contours, -1, (0, 255, 0), 3)
            discrete_err = 1
            # break
        # Ignore small contours
        # print(cv.contourArea(contour))
        # cv.imshow('image', img)
        # cv.waitKey(0)
        # print(cv.contourArea(contour))
        if cv.contourArea(contour) > 2500:
            # print('hello')

            # Draw a bounding box around the contour
            x, y, w, h = cv.boundingRect(contour)
            im_bw = change_outside_contour_to_white(im_bw, contour)
            average_w = w  # int((cv.contourArea(contour))/h)
            # average_h = int((cv.contourArea(contour))/w)
            # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get object dimensions and current distance in centimeters
            # print("Width: " + str(w) + " Height: " + str(h))
            cv.putText(
                img,
                f"Average_Width: {average_w:.2f} pixel",
                (0, y + 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            cv.putText(
                img,
                f"Average_Height: {h:.2f} pixel",
                (0, y + 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            # chia contour thành x phần bằng nhau
            colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
            ]
            # print(f'chieu dai la {h}, chieu dai chuan la {length}')
            if h > length + 10:
                # print(f'chieu dai la {h}, chieu dai chuan la {length}')
                l_error_big = l_error_big + 1
                # cv.putText(img, 'Thua chieu dai', (0, y + 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0  ), 1)
            if h < length - 28:
                # print(f'chieu dai la {h}, chieu dai chuan la {length}')
                l_error_small = l_error_small + 1
                cv.putText(
                    img,
                    "Thieu chieu dai",
                    (0, y + 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
            sections = divide_shape_by_bounding_height(
                contour, x=int(h / 2), image_shape=img.shape
            )
            count = 0
            for i, section in enumerate(sections):
                # cv.drawContours(img, [section], -1, colors[i % len(colors)], thickness=2)
                if count <= 5 or count >= len(sections) - 5:
                    count = count + 1
                    continue
                count = count + 1
                x, y, w, h = cv.boundingRect(section)
                section_width = w  # int((cv.contourArea(section))/h)
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # print(
                #     f"chieu rong cua section {i} la {section_width} pixel"
                # )  # dùng để debug
                # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if section_width > width + forgive_high:
                    # cv.rectangle(img, (x, y), (x + w, y + h),  (255, 0, 0), 2)
                    cv.drawContours(img, [section], -1, (255, 0, 0), thickness=1)
                    w_error_big = w_error_big + 1
                if section_width < width - forgive_low:
                    # cv.rectangle(img, (x, y), (x + w, y + h),  (0, 0, 255), 2)
                    cv.drawContours(img, [section], -1, (0, 0, 255), thickness=1)
                    w_error_small = w_error_small + 1
    if w_error_big != 0:
        print("Thừa chiều rộng")
    if w_error_small != 0:
        print("Thiếu chiều rộng")
    if l_error_big != 0:
        print("Thừa chiều dài")
    if l_error_small != 0:
        print("Thiếu chiều dài")
    if contournum > 2:
        print("Đứt đoạn")
    # cv.rectangle(img, (x, y), (x + w, y + h),  colors[i % len(colors)], 2)
    # cv.imshow("image", img)
    # cv.waitKey(0)
    return img, (w_error_big, w_error_small, l_error_big, l_error_small, discrete_err)


class GiaoDien:
    def __init__(self, MainWindow):
        self.main_window = MainWindow
        self.setupUi()

    def setupUi(self):
        self.main_window.setObjectName("MainWindow")
        self.main_window.resize(1360, int(1360 / 1.5))
        self.centralwidget = QtWidgets.QWidget(self.main_window)
        self.centralwidget.setObjectName("centralwidget")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(400, 30, 650, 91))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.labelCamRealTime = QtWidgets.QLabel(self.centralwidget)
        self.labelCamRealTime.setGeometry(
            QtCore.QRect(20, 100, 1000, (int)(1000 / 1.5))
        )
        self.labelCamRealTime.setObjectName("labelCamRealTime")
        self.labelCamRealTime.setStyleSheet("background-color: lightgray;")

        self.labelRecapFrame = QtWidgets.QLabel(self.centralwidget)
        self.labelRecapFrame.setGeometry(QtCore.QRect(1040, 100, 300, 200))
        self.labelRecapFrame.setObjectName("labelRecapFrame")
        self.labelRecapFrame.setStyleSheet("background-color: lightgray;")

        self.labelCroppedImage = QtWidgets.QLabel(self.centralwidget)
        self.labelCroppedImage.setGeometry(QtCore.QRect(1040, 320, 300, 200))
        self.labelCroppedImage.setObjectName("labelCroppedImage")
        self.labelCroppedImage.setStyleSheet("background-color: lightgray;")

        self.labelProcessedImage = QtWidgets.QLabel(self.centralwidget)
        self.labelProcessedImage.setGeometry(QtCore.QRect(1040, 540, 300, 200))
        self.labelProcessedImage.setObjectName("labelProcessedImage")
        self.labelProcessedImage.setStyleSheet("background-color: lightgray;")

        self.recapButton = QtWidgets.QPushButton(self.centralwidget)
        self.recapButton.setGeometry(QtCore.QRect(1040, 760, 100, 40))
        self.recapButton.setText("Recap")
        self.recapButton.setObjectName("recapButton")
        self.recapButton.clicked.connect(self.manual_recap)

        self.main_window.setCentralWidget(self.centralwidget)

        self.retranslateUi()

        self.camera_thread = threading.Thread(target=camera_thread, daemon=True)
        self.camera_thread.start()

        self.update_ui()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.main_window.setWindowTitle(
            _translate("MainWindow", "HỆ THỐNG PHÁT HIỆN LỖI DÁN KEO")
        )
        self.label.setText(_translate("MainWindow", "HỆ THỐNG PHÁT HIỆN LỖI DÁN KEO"))

    def update_ui(self):
        global frame, recap_frame, highest_sharpness
        global errs
        if frame is not None:
            resized_frame = cv.resize(frame, (1000, 750))
            # resized_frame = cv.resize(frame, (1920, 1440))
            img = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
            img = QImage(
                img.data,
                img.shape[1],
                img.shape[0],
                img.strides[0],
                QImage.Format_RGB888,
            )
            self.labelCamRealTime.setPixmap(QPixmap.fromImage(img))  # đưa ảnh vào label

            center_part = (0, 0)

            cropped, center_part = get_Cropped_image(image=frame)
            if cropped is not None:
                print(
                    "detect_object_center",
                    detect_object_center(
                        center=center_part, up_border=100, low_border=400
                    ),
                )

            if detect_object_center(center=center_part, up_border=100, low_border=400):
                sharpness = calculate_sharpness(frame)
                if sharpness > highest_sharpness:
                    highest_sharpness = sharpness
                    recap_frame = frame.copy()
            else:
                highest_sharpness = 0
                if recap_frame is not None:
                    status = True
                    for err in errs:
                        if err != 0:
                            status = False
                    if status:
                        arduino.send_command("1")  # Gửi lệnh "01" đến Arduino
                        print("opening")
                    else:
                        arduino.send_command("0")  # Gửi lệnh "00" đến Arduino
                        print("closing")

                    response = arduino.read_response()
                    if response:
                        print(f"Phản hồi từ Arduino: {response}")

            if recap_frame is not None:
                recap_resized = cv.resize(recap_frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
                recap_img = cv.cvtColor(recap_resized, cv.COLOR_BGR2RGB)
                recap_img = QImage(
                    recap_img.data,
                    recap_img.shape[1],
                    recap_img.shape[0],
                    recap_img.strides[0],
                    QImage.Format_RGB888,
                )
                self.labelRecapFrame.setPixmap(QPixmap.fromImage(recap_img))

                # timestamp = time.strftime("%Y%m%d_%H%M%S")
                # self.labelRecapFrame.setPixmap(QPixmap.fromImage(recap_img))
                # filename = os.path.join(RECAP_FOLDER, f"recapp_{timestamp}.jpg") #
                # cv.imwrite(filename, recap_img)
                # print(f"Saved recap frame to {filename}")

                # cv.imshow("Recap Frame", recap_frame)
                # cv.waitKey(0)
                # print("recap_frame", recap_frame.shape)

                # Crop ảnh từ recap frame
                cropped_image, _ = get_Cropped_image(recap_frame.copy())

                if cropped_image is not None:
                    cropped_resized = cv.resize(cropped_image, (300, 200))
                    cropped_img = cv.cvtColor(cropped_resized, cv.COLOR_BGR2RGB)
                    cropped_img = QImage(
                        cropped_img.data,
                        cropped_img.shape[1],
                        cropped_img.shape[0],
                        cropped_img.strides[0],
                        QImage.Format_RGB888,
                    )
                    self.labelCroppedImage.setPixmap(QPixmap.fromImage(cropped_img))

                    # Xử lý phát hiện lỗi
                    processed_image, errs = detect_error(recap_frame.copy())

                    result_resized = cv.resize(processed_image, (300, 200))
                    result_img = cv.cvtColor(result_resized, cv.COLOR_BGR2RGB)
                    result_img = QImage(
                        result_img.data,
                        result_img.shape[1],
                        result_img.shape[0],
                        result_img.strides[0],
                        QImage.Format_RGB888,
                    )
                    self.labelProcessedImage.setPixmap(QPixmap.fromImage(result_img))

        QtCore.QTimer.singleShot(10, self.update_ui)

    def manual_recap(self):
        global frame, recap_frame, highest_sharpness
        if frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(RECAP_FOLDER, f"rrrecap_{timestamp}.jpg")  #
            cv.imwrite(filename, frame)
            print(f"Saved recap frame to {filename}")

            recap_frame = frame.copy()
            highest_sharpness = calculate_sharpness(frame)
        # Lặp lại sau 10ms
        QtCore.QTimer.singleShot(10, self.update_ui)


arduino = ArduinoSerial(port="COM4", baudrate=9600)
arduino.connect()
app = QtWidgets.QApplication([])
window = QtWidgets.QMainWindow()
ui = GiaoDien(window)
window.show()
app.exec_()

cap.release()
cv.destroyAllWindows()
arduino.close()
