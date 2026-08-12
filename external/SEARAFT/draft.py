import cv2
import numpy as np
from numpy.linalg import norm

# --- 1. HÀM TÍNH TOÁN FOE VÀ LỌC NGOẠI LỆ ---

def estimate_foe_and_outliers(prev_pts, curr_pts, flow_vectors, shape):
    """
    Ước tính Tâm Mở rộng (FoE) bằng RANSAC và xác định các vector Flow ngoại lệ (outliers)
    đại diện cho đối tượng chuyển động độc lập.

    Args:
        prev_pts (np.array): Vị trí các điểm đặc trưng (x, y) trong Frame t.
        curr_pts (np.array): Vị trí các điểm đặc trưng được theo dõi trong Frame t+1.
        flow_vectors (np.array): Vector Flow 2D [u, v] tương ứng với từng cặp điểm.

    Returns:
        tuple: (foe_location, moving_mask)
               foe_location: Tọa độ (x, y) của FoE.
               moving_mask: Mask nhị phân (1D) đánh dấu các điểm outlier (chuyển động).
    """
    if len(prev_pts) < 4:
        # Cần tối thiểu 4 điểm để ước tính mô hình
        return None, np.zeros(len(prev_pts), dtype=bool)

    # FoE (Focus of Expansion) là điểm hội tụ của các vector Flow nền tĩnh.
    # Trong OpenCV, việc ước tính FoE thường được thực hiện thông qua mô hình chuyển động 
    # phức tạp hơn hoặc cần triển khai RANSAC tùy chỉnh.
    # Tuy nhiên, chúng ta có thể sử dụng RANSAC/findHomography để tìm ma trận M
    # (Homography) và các outliers là một cách xấp xỉ ban đầu để lọc chuyển động.
    # *LƯU Ý:* Để đạt được FoE chính xác, cần một thuật toán FoE RANSAC chuyên biệt .

    # Thay thế FoE RANSAC chuyên biệt bằng RANSAC Homography để lọc outliers thô:
    # Mục tiêu vẫn là tìm mask_outliers (chuyển động độc lập) 
    H, mask_inliers = cv2.findHomography(
        prev_pts, curr_pts, cv2.RANSAC, ransacReprojThreshold=3.0 # [1]
    )
    
    # Mask ngoại lệ (outliers) là các vector Flow KHÔNG thuộc về mô hình nền H
    # -> Đây là các vector ứng viên của đối tượng di chuyển.
    moving_mask = (mask_inliers.flatten() == 0)

    # Trong một triển khai FoE-based thực tế (ví dụ: FoELS ):
    # 1. RANSAC được dùng để tìm FoE từ các vector Flow có hướng nhất quán .
    # 2. FoE được xác định là Tâm (Source/Sink) của các Flow vector tĩnh .
    # 3. Moving Mask được tạo từ các điểm KHÔNG hướng về FoE hoặc có độ lớn không khớp.[2]

    # Do giới hạn thư viện OpenCV tiêu chuẩn, ta trả về H làm giả định chuyển động nền
    # và sử dụng outliers của H làm mask chuyển động ban đầu.
    # FoE_location thực tế chỉ có ý nghĩa khi camera tịnh tiến.
    foe_location = (shape[0] / 2, shape[1] / 2) # Giả định FoE nằm ở trung tâm ảnh nếu camera tịnh tiến thuần

    return foe_location, moving_mask


# --- 2. QUY TRÌNH CHÍNH (SỬ DỤNG DENSE FLOW TỪ SEA-RAFT) ---

def foe_based_moving_object_detection(frame_t, frame_t_plus_1, F_estimated, shape_image):
    """
    Phát hiện đối tượng chuyển động bằng cách phân tích Flow Dày Đặc (Dense Flow)
    dựa trên mô hình Tâm Mở rộng (FoE) gián tiếp.

    Args:
        frame_t, frame_t_plus_1: Ảnh gốc.
        F_estimated (np.array): Flow dày đặc [u, v] từ SEA-RAFT (H, W, 2).

    Returns:
        np.array: Mặt nạ nhị phân của đối tượng chuyển động.
    """
    height, width = frame_t.shape[:2]

    # BƯỚC 1: TRÍCH XUẤT VÀ THEO DÕI CÁC ĐIỂM ĐẶC TRƯNG (Sparse Flow)
    # Tương tự như GMC, ta cần một tập hợp các điểm để ước tính mô hình nền
    prev_gray = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame_t_plus_1, cv2.COLOR_BGR2GRAY)

    prev_pts_sparse = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=3)
    if prev_pts_sparse is None:
        return np.zeros((height, width), dtype=np.uint8)

    curr_pts_sparse, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts_sparse, None, winSize=(15, 15))
    
    # Lọc các điểm theo dõi thành công và chuẩn bị vector Flow
    status = status.flatten()
    prev_pts_f = prev_pts_sparse[status == 1]
    curr_pts_f = curr_pts_sparse[status == 1]
    
    # Chuyển đổi tọa độ về dạng 2D
    prev_pts_2d = prev_pts_f.reshape(-1, 2)
    curr_pts_2d = curr_pts_f.reshape(-1, 2)
    flow_sparse = curr_pts_2d - prev_pts_2d

    # BƯỚC 2: ƯỚC TÍNH FOE VÀ TẠO MẶT NẠ CHUYỂN ĐỘNG THÔ (dựa trên Outliers)
    # Sử dụng hàm FoE ước tính (hoặc RANSAC Homography proxy)
    foe_loc, moving_mask_sparse = estimate_foe_and_outliers(
        prev_pts_2d, curr_pts_2d, flow_sparse, shape_image
    )

    # BƯỚC 3: MỞ RỘNG MẶT NẠ THƯA SANG MẶT NẠ DÀY ĐẶC BẰNG DENSE FLOW (SEA-RAFT)
    
    # Tạo bản đồ độ lớn Flow Dày Đặc
    flow_u, flow_v = F_estimated[:, :, 0], F_estimated[:, :, 1]
    V_estimated = norm(F_estimated, axis=2) # Độ lớn Flow thực tế [3]
    
    # Áp dụng ngưỡng trên độ lớn Flow (dễ bị nhiễu do FoE/Parallax)
    THRESHOLD_RAW = 1.5 # Ngưỡng Flow cơ bản

    # BƯỚC 4: HỢP NHẤT (Fusion) Saliency Dày Đặc và Mask Thưa của FoE 
    
    # Trong các phương pháp FoE tiên tiến (FoELS), Moving Pixel Likelihood được tính toán
    # từ các outliers FoE . Ta mô phỏng bằng cách:

    # 4.1 Tăng cường tín hiệu Flow Dày Đặc:
    # Nếu một vùng có Flow lớn V_estimated, nhưng lại nằm gần các điểm outliers (moving_mask_sparse)
    # thì nó có khả năng là đối tượng di chuyển.

    # Lấy các điểm ngoại lệ (outliers) đã được FoE/RANSAC xác nhận
    outlier_pts = prev_pts_2d[moving_mask_sparse]
    
    # Tạo một mặt nạ Saliency Dày Đặc (Saliency Map) bằng cách làm mờ (Blurring)
    # hoặc sử dụng Distance Transform từ các điểm outlier để xác định vùng ảnh hưởng.
    
    # Ví dụ đơn giản: Chỉ lấy Flow lớn VÀ gần một outlier đã biết
    
    # Khởi tạo Moving Object Mask (M_final)
    M_final = np.zeros((height, width), dtype=np.uint8)
    
    # Đối với mỗi điểm ngoại lệ (outlier point) đã xác định:
    for x, y in outlier_pts:
        # Sử dụng V_estimated (Flow dày đặc SEA-RAFT) để mở rộng vùng chuyển động [4, 3]
        # Ta xem xét một vùng lân cận quanh outlier (ví dụ: bán kính 30 pixel)
        x, y = int(x), int(y)
        
        # Cắt xén vùng ảnh hưởng (đảm bảo không vượt quá giới hạn ảnh)
        min_r, max_r = max(0, y - 30), min(height, y + 30)
        min_c, max_c = max(0, x - 30), min(width, x + 30)
        
        # Áp dụng ngưỡng cục bộ của SEA-RAFT Flow trong vùng này
        # (Điều kiện: V_estimated phải đủ lớn VÀ vùng này được FoE xác nhận là outlier)
        local_motion = (V_estimated[min_r:max_r, min_c:max_c] > THRESHOLD_RAW).astype(np.uint8)
        
        # Hợp nhất vào mặt nạ cuối cùng
        M_final[min_r:max_r, min_c:max_c] = cv2.bitwise_or(M_final[min_r:max_r, min_c:max_c], local_motion)

    # BƯỚC 5: HẬU XỬ LÝ HÌNH THÁI HỌC (Làm sạch và làm mịn)
    kernel = np.ones((5, 5), np.uint8)
    M_final = cv2.morphologyEx(M_final, cv2.MORPH_OPEN, kernel) # Erosion sau đó Dilation (Mở) [3]
    M_final = cv2.dilate(M_final, kernel, iterations=1) # Làm giãn để lấp đầy lỗ hổng

    # BƯỚC 6: TRÍCH XUẤT BOUNDING BOX (Tùy chọn)
    contours, _ = cv2.findContours(M_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        print("x, y, w, h : ", x, y, w, h)
        
        boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])
        
        # # if w*h > min_area and w < max_size and h < max_size and x < pos_limit and y < pos_limit and x < img_size[1] - pos_limit and y < img_size[0] - pos_limit:  # bỏ qua các bbox quá nhỏ
        # if w*h >= min_area and w < max_size and h < max_size and w > 5 and h > 5 : # bỏ qua các bbox quá nhỏ
        # # if w*h >= min_area and w < max_size and h < max_size : # bỏ qua các bbox quá nhỏ
        #     # print("add old x, y, w, h : ", x, y, w, h)
        #     boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])
            
        # if w > max_size or y > max_size :
        #     x = input()
        
        # if w*h > min_area and w < max_size and h < max_size and x > pos_limit and y > pos_limit and x < (img_size[1] - pos_limit) and y < (img_size[0] - pos_limit):  # bỏ qua các bbox quá nhỏ
        # # if w*h > min_area and w < max_size and h < max_size :  # bỏ qua các bbox quá nhỏ
        #     print("add x, y, w, h : ", x, y, w, h)
        #     boxes.append([x  - w/2,y - h/2,x + w * 1.5, y + h * 1.5 ,1])


    print("M_final : ", M_final)
    print("max : ", np.max(M_final))

    return M_final.astype(np.uint8) * 255, boxes