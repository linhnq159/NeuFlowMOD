"""Shared RGB preprocessing and past-to-current camera-motion compensation."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
import torch


STABILIZATION_MODES = ('none', 'homography')
MAX_ESTIMATION_SIDE = 960
MIN_INLIERS = 12
MIN_INLIER_RATIO = 0.5
MIN_SPATIAL_COVERAGE = 0.05
MIN_OVERLAP = 0.5


def init_frame_worker(worker_id):
    # DataLoader already parallelizes pairs; avoid an OpenCV thread pool per worker.
    cv2.setNumThreads(1)


@dataclass
class AlignmentInfo:
    mode: str
    success: bool = False
    reason: str = 'disabled'
    matches: int = 0
    inliers: int = 0
    overlap: float = 0.0
    homography: np.ndarray | None = None

    def summary(self):
        status = 'aligned' if self.success else ('disabled' if self.mode == 'none' else 'fallback')
        return (f'Stabilization: {status}; reason={self.reason}; '
                f'matches={self.matches}; inliers={self.inliers}; overlap={self.overlap:.3f}')


def _ratio_matches(descriptors0, descriptors1):
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors0, descriptors1, k=2)
    return {pair[0].queryIdx: pair[0].trainIdx for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance}


def _valid_geometry(homography, width, height):
    """Reject singular transforms, horizon crossings, reflection and folded quads."""
    if homography is None or homography.shape != (3, 3) or not np.isfinite(homography).all():
        return False
    magnitude = np.max(np.abs(homography))
    if magnitude == 0 or np.linalg.matrix_rank(homography / magnitude) < 3:
        return False
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                        [0, height - 1]], dtype=np.float64)
    projected = np.column_stack((corners, np.ones(4))) @ homography.T
    denominator = projected[:, 2]
    tolerance = magnitude * 1e-10
    if not (np.all(denominator > tolerance) or np.all(denominator < -tolerance)):
        return False
    points = projected[:, :2] / denominator[:, None]
    if not np.isfinite(points).all() or np.max(np.abs(points)) > np.finfo(np.float32).max:
        return False
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    return bool(cv2.isContourConvex(contour) and cv2.contourArea(contour, oriented=True) > 0)


def warp_past_to_current(current, past, homography):
    """Warp only past; fill pixels without a full interpolation footprint from current."""
    height, width = current.shape[:2]
    aligned = cv2.warpPerspective(past, homography, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    coverage = cv2.warpPerspective(np.ones(past.shape[:2], dtype=np.float32), homography,
                                   (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = coverage >= 1.0 - 1e-6
    aligned[~valid] = current[~valid]
    return aligned, valid


def stabilize_past(current, past):
    """Estimate a global homography from RGB uint8 frames, without segmentation labels."""
    if current.shape != past.shape or current.ndim != 3 or current.shape[2] != 3:
        raise ValueError('Expected matching RGB frame arrays [H, W, 3].')
    info = AlignmentInfo(mode='homography', reason='insufficient_features')
    height, width = current.shape[:2]
    scale = min(1.0, MAX_ESTIMATION_SIDE / max(height, width))
    small_width, small_height = max(1, round(width * scale)), max(1, round(height * scale))
    try:
        gray_current = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        gray_past = cv2.cvtColor(past, cv2.COLOR_RGB2GRAY)
        if scale < 1:
            gray_current = cv2.resize(gray_current, (small_width, small_height), interpolation=cv2.INTER_AREA)
            gray_past = cv2.resize(gray_past, (small_width, small_height), interpolation=cv2.INTER_AREA)
        detector = cv2.SIFT_create(nfeatures=2000)
        past_points, past_descriptors = detector.detectAndCompute(gray_past, None)
        current_points, current_descriptors = detector.detectAndCompute(gray_current, None)
        if (past_descriptors is None or current_descriptors is None
                or min(len(past_descriptors), len(current_descriptors)) < MIN_INLIERS):
            return past, info
        forward = _ratio_matches(past_descriptors, current_descriptors)
        backward = _ratio_matches(current_descriptors, past_descriptors)
        matches = [(a, b) for a, b in forward.items() if backward.get(b) == a]
        info.matches = len(matches)
        info.reason = 'insufficient_matches'
        if info.matches < MIN_INLIERS:
            return past, info
        source = np.float32([past_points[a].pt for a, _ in matches])
        target = np.float32([current_points[b].pt for _, b in matches])
        homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0,
                                                maxIters=2000, confidence=0.995)
        info.reason = 'insufficient_inliers'
        if inliers is None:
            return past, info
        selected = inliers.ravel().astype(bool)
        info.inliers = int(selected.sum())
        if info.inliers < MIN_INLIERS or info.inliers / info.matches < MIN_INLIER_RATIO:
            return past, info
        info.reason = 'localized_inliers'
        for points in (source[selected], target[selected]):
            area = cv2.contourArea(cv2.convexHull(points))
            if area / (small_width * small_height) < MIN_SPATIAL_COVERAGE:
                return past, info
        info.reason = 'invalid_homography'
        if not _valid_geometry(homography, small_width, small_height):
            return past, info
        # OpenCV resize maps pixel centers by x_small = scale * (x + .5) - .5.
        sx, sy = small_width / width, small_height / height
        resize_transform = np.array([[sx, 0, (sx - 1) / 2],
                                     [0, sy, (sy - 1) / 2], [0, 0, 1]])
        homography = np.linalg.inv(resize_transform) @ homography @ resize_transform
        if not _valid_geometry(homography, width, height):
            return past, info
        aligned, valid = warp_past_to_current(current, past, homography)
        info.overlap = float(valid.mean())
        info.reason = 'insufficient_overlap'
        if info.overlap < MIN_OVERLAP:
            return past, info
        info.success, info.reason, info.homography = True, 'ok', homography
        return aligned, info
    except (cv2.error, np.linalg.LinAlgError):
        info.reason = 'estimation_error'
        return past, info


def prepare_frame_pair(current, past, img_size=(960, 544), stabilization='none'):
    """PIL frames -> current/past float RGB tensors and alignment diagnostics."""
    if stabilization not in STABILIZATION_MODES:
        raise ValueError(f'Unsupported stabilization: {stabilization}')
    if current.size != past.size:
        raise ValueError(f'Frame sizes differ: current={current.size}, past={past.size}')
    if len(img_size) != 2 or any(not isinstance(n, int) or n < 1 for n in img_size):
        raise ValueError('img_size must contain positive integer WIDTH HEIGHT values.')
    current_rgb = np.array(current.convert('RGB').resize(tuple(img_size), Image.Resampling.BILINEAR))
    past_rgb = np.array(past.convert('RGB').resize(tuple(img_size), Image.Resampling.BILINEAR))
    info = AlignmentInfo(mode='none')
    if stabilization == 'homography':
        past_rgb, info = stabilize_past(current_rgb, past_rgb)
    current_tensor = torch.from_numpy(current_rgb.astype(np.float32)).permute(2, 0, 1)
    past_tensor = torch.from_numpy(past_rgb.astype(np.float32)).permute(2, 0, 1)
    return current_tensor, past_tensor, info


def load_frame_pair(current_path, past_path, img_size=(960, 544), stabilization='none'):
    with Image.open(current_path) as current, Image.open(past_path) as past:
        tensors = prepare_frame_pair(current, past, img_size, stabilization)
        return (*tensors, current.size)
