import cv2
import numpy as np
from tools.do_kiem_1_copy import run_optical_flow, stabilize_image, tensor_to_flow_numpy, run_detect_object

# Giả định: Bạn đã có 2 file ảnh trong thư mục
# image_1.jpg: Khung hình 1 (Nền tham chiếu)
# image_2.jpg: Khung hình 2 (Khung hình hiện tại)
IMAGE_PATH_1 = '/mnt/data/linhnq11/Data/moving_object_detection/IR_bia_TANK/IR_bia_TANK0002/IR_bia_TANK0002_10700.jpg'
IMAGE_PATH_2 = '/mnt/data/linhnq11/Data/moving_object_detection/IR_bia_TANK/IR_bia_TANK0002/IR_bia_TANK0002_10710.jpg'

# 1. Tải 2 hình ảnh (frame) đầu vào
frame_1 = cv2.imread(IMAGE_PATH_1)
frame_2 = cv2.imread(IMAGE_PATH_2)

image1_stable, matrix_stable = stabilize_image(frame_2, frame_1, method='homography')

if image1_stable is None or frame_2 is None:
    print("Lỗi: Không tìm thấy file ảnh hoặc đường dẫn sai.")
    print("Vui lòng thay đổi 'image_1.jpg' và 'image_2.jpg' bằng đường dẫn file ảnh thật của bạn.")
else:
    # Đảm bảo 2 ảnh có cùng kích thước
    if image1_stable.shape != frame_2.shape:
        # Nếu không, resize frame 2 về kích thước của frame 1
        frame_2 = cv2.resize(frame_2, (image1_stable.shape[1], image1_stable.shape[0]))
    
    # 2. Chuyển sang ảnh grayscale (đen trắng) để dễ tính toán
    gray_1 = cv2.cvtColor(image1_stable, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    
    # 3. Áp dụng Frame Difference (Trừ khung hình)
    # Tính giá trị tuyệt đối của sự khác biệt giữa hai frame.
    # Các pixel có sự thay đổi lớn sẽ có giá trị cao.
    frame_diff = cv2.absdiff(gray_1, gray_2)

    # 4. Áp dụng Ngưỡng (Thresholding)
    # Đặt ngưỡng (ví dụ: 30). Nếu sự khác biệt lớn hơn 30, coi đó là chuyển động (255 - trắng).
    # THRESH_BINARY_INV để đảo ngược, nhưng ở đây ta dùng THRESH_BINARY để lấy vùng chuyển động màu trắng.
    thresh_value = 30
    _, thresh_mask = cv2.threshold(frame_diff, thresh_value, 255, cv2.THRESH_BINARY)
    
    # 5. Xử lý hậu kỳ (Morphology)
    # Loại bỏ nhiễu pixel nhỏ bằng phép 'Dilate' (Giãn nở)
    kernel = np.ones((5,5),np.uint8)
    final_mask = cv2.dilate(thresh_mask, kernel, iterations=2)

    # 6. Hiển thị kết quả
    cv2.imshow("Frame 1 (Nền)", image1_stable)
    cv2.imshow("Frame 2 (Hiện tại)", frame_2)
    cv2.imshow("Absolute Difference", frame_diff)
    cv2.imshow("Motion Detected (Trắng)", final_mask)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()