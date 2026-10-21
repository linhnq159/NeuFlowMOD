"""Contour-based bounding boxes for original-size binary segmentation masks."""

import math

import cv2
import numpy as np


def add_box_arguments(parser):
    parser.add_argument('--draw_boxes', action='store_true',
                        help='Find external mask contours and draw red bounding boxes on the current frame')
    parser.add_argument('--min_contour_area', type=float, default=0.,
                        help='Minimum OpenCV contour area in original-image pixel units; default 0 keeps tiny regions')
    parser.add_argument('--box_thickness', type=int, default=2,
                        help='Bounding-box line thickness in original-image pixels (default: 2)')


def validate_box_arguments(args, parser):
    if not math.isfinite(args.min_contour_area) or args.min_contour_area < 0:
        parser.error('--min_contour_area must be finite and >= 0.')
    if args.box_thickness < 1:
        parser.error('--box_thickness must be >= 1.')


def contour_boxes(mask, min_contour_area=0.):
    """Return external contour boxes as sorted [x, y, width, height] lists.

    Area is OpenCV's geometric contour area (not foreground-pixel count).
    Single pixels/lines have zero contour area and are retained at the default.
    The caller's binary mask is never changed.
    """
    if mask.ndim != 2:
        raise ValueError('Expected a 2D segmentation mask.')
    binary = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [list(cv2.boundingRect(contour)) for contour in contours
             if cv2.contourArea(contour) >= min_contour_area]
    return sorted(boxes, key=lambda box: (box[1], box[0], box[2], box[3]))


def draw_bounding_boxes(rgb, boxes, thickness=2):
    """Draw red rectangles on a copy of an RGB image, using inclusive corners."""
    result = np.asarray(rgb).copy()
    for x, y, width, height in boxes:
        cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1),
                      (255, 0, 0), thickness, lineType=cv2.LINE_8)
    return result
