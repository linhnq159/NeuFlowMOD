import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

class StabTest:
    def __init__(self):
        self.pre_img = None
        self.detector = cv2.ORB_create()
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        self.gme_rs = np.eye(3, dtype=np.float32)
        self.gme = self.gme_rs.copy()
        self.delta_gme = np.array([0, 0])
        self.center_pos = np.array([640, 360])  # image center position for 1280x720 image

    def init(self, in_frame):
        print('----StabTest init---')
        # self.center_pos = np.array([in_frame.shape[1]//2, in_frame.shape[0]//2])
        self.center_pos = np.array([331, 247])
        self.pre_img = in_frame.copy()
        self.gme_rs = np.eye(3, dtype=np.float32)
        return True

    def process(self, in_frame):
        frame = in_frame.copy()
        print('----StabTest process---')
        # print('pre_img vs frame SSIM', ssim(self.pre_img[:,:,0], frame[:,:,0]))
        kp1, des1 = self.detector.detectAndCompute(self.pre_img, None)
        kp2, des2 = self.detector.detectAndCompute(frame, None)
        # print('kp1', kp1)
        # print('kp2', kp2)
        # print('des1', des1)
        # print('des2', des2)
        if des1 is None or des2 is None:
            self.gme_rs = np.eye(3, dtype=np.float32)
            return self.gme_rs

        matches = self.matcher.match(des1, des2, None)
        matches = sorted(matches, key=lambda x: x.distance)
        # print('number matches', len(matches))
        if len(matches) < 4:
            self.gme_rs = np.eye(3, dtype=np.float32)
            return self.gme_rs

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:30]]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:30]]).reshape(-1, 1, 2)

        self.gme_rs, mask = cv2.estimateAffinePartial2D(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=2)

        if self.gme_rs is None:
            self.gme_rs = np.eye(3, dtype=np.float32)
        else:
            self.gme_rs = np.vstack([self.gme_rs, [0, 0, 1]])

        # if not (pts1==pts2).all():
        #     print('pts1', pts1)
        #     print('pts2', pts2)
        #     print('pts1==pts2', (pts1==pts2).all())
        #     print('gme_rs', self.gme_rs)
        #     exit()
        self.pre_img = frame.copy()
        return self.gme_rs

    def get_gme_matrix(self):
        """
        Trả về ma trận 3x3 (không transpose).
        Tính center_pos mới bằng phép nhân homography đúng chuẩn.
        """
        if self.gme_rs is None or self.gme_rs.size == 0:
            return None

        # self.gme_rs hiện đã là 3x3 (từ process)
        H = self.gme_rs.copy().astype(np.float64)  # giữ nguyên, không transpose

        # Normalize để phần tử H[2,2] = 1 (tránh scaling lạ)
        if abs(H[2,2]) > 1e-12:
            H = H / H[2,2]
        else:
            H[2,2] = 1.0

        # compute center position in homogeneous coords: p' = H * [cx, cy, 1]^T
        cx, cy = float(self.center_pos[0]), float(self.center_pos[1])
        p_h = H @ np.array([cx, cy, 1.0], dtype=np.float64)  # shape (3,)
        if abs(p_h[2]) < 1e-12:
            # guard divide-by-zero
            new_cx, new_cy = p_h[0], p_h[1]
        else:
            new_cx, new_cy = float(p_h[0]/p_h[2]), float(p_h[1]/p_h[2])

        center_pos_new = np.array([new_cx, new_cy], dtype=np.float64)

        # compute delta (dịch chuyển) — bạn đang tích luỹ delta_gme
        dx = center_pos_new[0] - cx
        dy = center_pos_new[1] - cy
        self.delta_gme[0] += dx
        self.delta_gme[1] += dy

        # update center_pos
        self.center_pos = center_pos_new

        # Trả về ma trận H ở dạng float32 (dùng cho warpPerspective) và meta
        return H.astype(np.float32), {'center_old': (cx, cy), 'center_new': (new_cx, new_cy), 'delta': (dx, dy)}

    # Usage example:
    # stab_test = StabTest()
    # frame = cv2.imread("path_to_image")
    # stab_test.init(frame)
    # result = stab_test.process(frame)
    # gme_matrix = stab_test.get_gme_matrix()