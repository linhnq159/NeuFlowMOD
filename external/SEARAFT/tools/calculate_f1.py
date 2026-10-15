import os
import xml.etree.ElementTree as ET
import numpy as np

def iou(box1, box2):
    x1 = box1[0]
    y1 = box1[1]
    w1 = box1[2]
    h1 = box1[3]
    
    x2 = box2[0]
    y2 = box2[1]
    w2 = box2[2]
    h2 = box2[3]
    
    # Calculate the coordinates of the intersection rectangle
    intersection_x1 = max(x1, x2)
    intersection_y1 = max(y1, y2)
    intersection_x2 = min(x1 + w1, x2 + w2)
    intersection_y2 = min(y1 + h1, y2 + h2)

    # Calculate area of intersection rectangle
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

    # Calculate area of union rectangle
    union_area = w1 * h1 + w2 * h2 - intersection_area

    # Calculate IOU
    iou = intersection_area / union_area
    
    if union_area <= 0:
        return 0.0
    
    return iou

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return [x1, y1, x2, y2]

def _match_predictions_get_flags(all_gts, preds_sorted, iou_threshold, iou_func):
    """
    Match predictions to GTs and return list of flags per prediction:
      tp_flags: list of 1/0 (1 = TP, 0 = FP) in the same order as preds_sorted
    Also returns total_gts (int).
    all_gts: dict image_id -> list of gt boxes (format consistent with iou_func)
    preds_sorted: list of {'image_id':..., 'bbox':..., 'score':...} sorted by score desc
    iou_func: function(box_pred, box_gt) -> float  (expects same box format)
    """
    # prepare matched flags per GT
    gt_matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in all_gts.items()}
    total_gts = sum(len(v) for v in all_gts.values())

    tp_flags = []
    for pred in preds_sorted:
        img_id = pred['image_id']
        pbox = pred['bbox']
        best_iou = 0.0
        best_idx = -1
        if img_id in all_gts:
            for idx, gt_box in enumerate(all_gts[img_id]):
                cur_iou = iou_func(pbox, gt_box)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = idx
        if best_iou >= iou_threshold and best_idx != -1 and (not gt_matched[img_id][best_idx]):
            tp_flags.append(1)
            gt_matched[img_id][best_idx] = True
        else:
            tp_flags.append(0)
    return tp_flags, total_gts

def compute_ap_from_predictions(all_gts, all_preds, iou_threshold=0.5, iou_func=iou):
    """
    Compute AP for a single IoU threshold.
    all_gts: dict image_id -> list of GT boxes (format matching all_preds' bbox format)
    all_preds: list of {'image_id':..., 'bbox':..., 'score':...}
    iou_threshold: float
    iou_func: function to compute IoU (expects bbox formats used in all_gts/all_preds)
    Returns:
      ap: float (Average Precision)
      precision: numpy array per detection
      recall: numpy array per detection
      precisions_interp: precision envelope (non-increasing)
      tp_flags: list of 1/0 per prediction (sorted by score desc)
      scores: list of scores (sorted desc)
    """
    if len(all_preds) == 0:
        return 0.0, np.array([]), np.array([]), np.array([]), [], []

    # sort preds by score descending
    preds_sorted = sorted(all_preds, key=lambda x: x['score'], reverse=True)

    # match predictions -> get tp flags per prediction
    tp_flags, total_gts = _match_predictions_get_flags(all_gts, preds_sorted, iou_threshold, iou_func)

    tp_cum = np.cumsum(tp_flags).astype(float)
    fp_flags = [1 - t for t in tp_flags]
    fp_cum = np.cumsum(fp_flags).astype(float)

    precisions = tp_cum / (tp_cum + fp_cum + 1e-12)
    recalls = tp_cum / total_gts if total_gts > 0 else np.zeros_like(precisions)

    # Make precision envelope (non-increasing)
    precisions_interp = precisions.copy()
    for i in range(len(precisions_interp)-2, -1, -1):
        if precisions_interp[i] < precisions_interp[i+1]:
            precisions_interp[i] = precisions_interp[i+1]

    # Compute AP = integral over recall of precision (use discrete sum over recall steps)
    # AP = sum_{i} (R_i - R_{i-1}) * P_i
    recall_prev = 0.0
    ap = 0.0
    for i in range(len(recalls)):
        delta_r = recalls[i] - recall_prev
        if delta_r > 0:
            ap += precisions_interp[i] * delta_r
            recall_prev = recalls[i]

    scores = [p['score'] for p in preds_sorted]
    return float(ap), precisions, recalls, precisions_interp, tp_flags, scores

def compute_map(all_gts, all_preds, iou_thresholds=[0.5], iou_func=iou):
    """
    Compute AP for each IoU threshold and return mAP (mean AP).
    all_gts: dict image_id -> list of GT boxes
    all_preds: list of {'image_id':..., 'bbox':..., 'score':...}
    iou_thresholds: list or array of IoU thresholds (e.g., [0.5] or np.linspace(0.5,0.95,10))
    iou_func: function to compute IoU
    Returns:
      result: dict {
        'aps': {t: ap, ...},
        'map': float,
        'details': {t: (ap, precisions, recalls, precisions_interp, tp_flags, scores)}
      }
    """
    aps = {}
    details = {}
    for t in iou_thresholds:
        ap, precisions, recalls, prec_env, tp_flags, scores = compute_ap_from_predictions(
            all_gts, all_preds, iou_threshold=t, iou_func=iou_func
        )
        aps[float(t)] = float(ap)
        details[float(t)] = (ap, precisions, recalls, prec_env, tp_flags, scores)

    mean_ap = float(np.mean(list(aps.values()))) if len(aps) > 0 else 0.0
    return {'aps': aps, 'map': mean_ap, 'details': details}

# -------------------------
# Load single VOC xml -> list of bboxes and size
# -------------------------
def parse_voc_xml(xml_path):
    """
    Parse a single Pascal VOC xml file.
    Returns: (bboxes_list, (width,height) or None)
    bboxes_list: [[x1,y1,x2,y2], ...] (float)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        raise RuntimeError(f"Cannot parse XML {xml_path}: {e}")

    # size
    width = None; height = None
    size_node = root.find('size')
    if size_node is not None:
        w_node = size_node.find('width')
        h_node = size_node.find('height')
        try:
            width = int(float(w_node.text)) if w_node is not None and w_node.text else None
            height = int(float(h_node.text)) if h_node is not None and h_node.text else None
        except:
            width = height = None

    bboxes = []
    for obj in root.findall('object'):
        bnd = obj.find('bndbox')
        if bnd is None:
            continue
        try:
            xmin = float(bnd.find('xmin').text)
            ymin = float(bnd.find('ymin').text)
            xmax = float(bnd.find('xmax').text)
            ymax = float(bnd.find('ymax').text)
        except:
            continue
        if (xmax - xmin) * (ymax - ymin) >= 36 and (xmax - xmin) < 100 and (ymax - ymin) < 100:
            bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
        else:
            print("(xmax - xmin) * (ymax - ymin) : ", (xmax - xmin) * (ymax - ymin))
        # bboxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
    return bboxes, (width, height) if width and height else None

# -------------------------
# Load predictions from a .txt file (bbox format or yolo)
# -------------------------
def load_pred_file_bbox(path):
    """
    Parse prediction file lines:
      x1 y1 x2 y2 [score]
    Returns list of {'bbox': [x1,y1,x2,y2], 'score': float}
    """
    preds = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                x1 = float(parts[2]); y1 = float(parts[3]); x2 = float(parts[4]); y2 = float(parts[5])
                score = float(parts[6]) if len(parts) >= 5 else 1.0
            except:
                continue
            preds.append({'bbox': [x1, y1, x2, y2], 'score': float(score)})
    
    return preds

def load_pred_file_yolo(path, img_size=(640,640)):
    """
    Parse YOLO normalized prediction file:
      class xc yc w h [score]
    Need img_size to convert -> absolute
    """
    preds = []
    img_w, img_h = img_size
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                xc = float(parts[1]); yc = float(parts[2]); w = float(parts[3]); h = float(parts[4])
                score = float(parts[5]) if len(parts) >= 6 else 1.0
            except:
                continue
            # bbox = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
            bbox = [xc, yc, w, h]
            preds.append({'bbox': bbox, 'score': float(score)})
    return preds

# -------------------------
# Main: walk GT root, find preds (matching relative path or basename fallback)
# -------------------------
def gather_dataset_pairs(gt_root, pred_root):
    """
    Walk recursively gt_root, return list of tuples:
      (image_id_rel, gt_xml_path, pred_txt_expected_relative_path)
    image_id_rel: relative path without extension, using '/' as separator for consistency.
    pred lookup strategy later will try:
      1) pred_root / relative_path + '.txt'
      2) pred_root / basename + '.txt' (fallback)
    """
    pairs = []
    for dirpath, _, filenames in os.walk(gt_root):
        for fname in filenames:
            if not fname.lower().endswith('.xml'):
                continue
            full = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full, gt_root)  # e.g. dir1/dir2/frame0001.xml
            rel_noext = os.path.splitext(rel_path)[0]
            # normalize separator to '/'
            image_id_rel = rel_noext.replace(os.sep, '/')
            pairs.append((image_id_rel, full))
    return pairs

def find_pred_file_for(image_id_rel, gt_xml_path, gt_root, pred_root):
    """
    Try to find prediction file for this image:
      1) pred_root / image_id_rel + '.txt'  (mirrored structure)
      2) pred_root / basename(image_id_rel) + '.txt'  (flat structure)
    If none exists, return None.
    """
    
    # 1) mirrored
    candidate1 = os.path.join(pred_root, image_id_rel.replace('annotations/', '').replace('/', os.sep) + '.txt')
    
    if os.path.isfile(candidate1):
        return candidate1
    # 2) basename
    basename = os.path.splitext(os.path.basename(image_id_rel))[0]
    candidate2 = os.path.join(pred_root, basename + '.txt')
    if os.path.isfile(candidate2):
        return candidate2
    # 3) try same directory as xml (maybe predictions saved alongside xml)
    candidate3 = os.path.splitext(gt_xml_path)[0] + '.txt'
    if os.path.isfile(candidate3):
        return candidate3
    return None

# -------------------------
# Evaluate whole dataset (many folders)
# -------------------------
def evaluate_dataset_many_folders(
    gt_root,
    pred_root,
    pred_format='bbox',   # 'bbox' or 'yolo'
    iou_threshold=0.5,
    default_img_size=(640,640),
    show_false_img = False,
    verbose=True
):
    """
    Evaluate dataset when GT XMLs are distributed in many subfolders under gt_root.
    pred_root may mirror structure or be flat.
    Returns result dict with dataset-level metrics.
    """
    pairs = gather_dataset_pairs(gt_root, pred_root)
    if len(pairs) == 0:
        raise RuntimeError("No GT xml files found under gt_root: " + str(gt_root))

    # containers aggregated
    all_gts = {}   # image_id_rel -> list of GT boxes
    all_preds = [] # list of {'image_id': id_rel, 'bbox':..., 'score':...}
    sizes = {}     # image_id_rel -> (w,h) or None

    for image_id_rel, xml_path in pairs:
        gts, size = parse_voc_xml(xml_path)
        all_gts[image_id_rel] = gts
        sizes[image_id_rel] = size

        # find pred file
        pred_file = find_pred_file_for(image_id_rel, xml_path, gt_root, pred_root)
        
        if pred_file is None:
            # no predictions for this image -> continue (will count FN later)
            continue
        
        # load preds according format
        if pred_format == 'bbox':
            preds_local = load_pred_file_bbox(pred_file)
        elif pred_format == 'yolo':
            img_size = size if size else default_img_size
            preds_local = load_pred_file_yolo(pred_file, img_size=img_size)
        else:
            raise ValueError("pred_format must be 'bbox' or 'yolo'")

        # append with image id
        for p in preds_local:
            all_preds.append({'image_id': image_id_rel, 'bbox': p['bbox'], 'score': p['score']})

    # Now we have all_gts and all_preds aggregated across dataset
    # Sort preds by score desc
    preds_sorted = sorted(all_preds, key=lambda x: x['image_id'], reverse=False)
    # preds_sorted = all_preds

    # prepare matched flags for GT
    gt_matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in all_gts.items()}

    TP = 0; FP = 0
    matched_ious = []
    total_gts = sum(len(v) for v in all_gts.values())

    for pred in preds_sorted:
        img_id = pred['image_id']
        pbox = pred['bbox']
        best_iou = 0.0; best_idx = -1
        if img_id in all_gts:
            for idx, gt_box in enumerate(all_gts[img_id]):
                # print("pbox : {}, gt_box : {}".format(pbox, gt_box))
                cur = iou(pbox, gt_box)
                # print("cur iou : ", cur)
                if cur > best_iou:
                    best_iou = cur; best_idx = idx
        # print("best_iou : ", best_iou)
        if best_iou >= iou_threshold and best_idx != -1 and (not gt_matched[img_id][best_idx]):
            # print("img_id : {}, pbox: {}, gt_box : {}".format(img_id, pbox, gt_box))
            TP += 1
            gt_matched[img_id][best_idx] = True
            matched_ious.append(best_iou)
        else:
            FP += 1

    FN = total_gts - TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if len(matched_ious) > 0 else 0.0
    mean_ji_percent = mean_iou * 100.0

    result = {
        'total_images': len(pairs),
        'total_gts': int(total_gts),
        'total_preds': len(all_preds),
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mean_iou': float(mean_iou),
        'mean_JI_percent': float(mean_ji_percent),
        'matched_ious': matched_ious,
        'mAP' : compute_map(all_gts, all_preds, iou_thresholds=[0, 0.5], iou_func=iou)['map']
    }

    if verbose:
        print("=== Dataset (many-folders) evaluation ===")
        print(f"GT xml files scanned: {len(pairs)}")
        print(f"Total GT boxes      : {result['total_gts']}")
        print(f"Total predictions   : {result['total_preds']}")
        print(f"TP / FP / FN        : {result['TP']} / {result['FP']} / {result['FN']}")
        print(f"Precision           : {result['precision']:.4f}")
        print(f"Recall              : {result['recall']:.4f}")
        print(f"F1-score            : {result['f1']:.4f}")
        print(f"Mean IoU (TPs)      : {result['mean_iou']:.4f}")
        print(f"JI (%)              : {result['mean_JI_percent']:.2f}% (mean over matched pairs)")
        # print(f"mAP@[0,0.5]         : {result['mAP']:.4f}")
        print("=========================================")

    return result

# -------------------------
# Example usage:
# -------------------------
if __name__ == "__main__":
    # Example:
    gt_root = "/mnt/data/linhnq11/moving_object_detection/ZBS/demo/PESMOD"   # contains many subfolders with .xml
    # pred_root = "/mnt/data/linhnq11/moving_object_detection/SEA-RAFT/output/do_kiem/iters12_M_spring_skip3frame/log" # may mirror structure or be flat
    pred_root = "/mnt/data/linhnq11/moving_object_detection/SEA-RAFT/output/ROI_draft/log" # may mirror structure or be flat
    # pred_root = "/mnt/data/linhnq11/moving_object_detection/SEA-RAFT/output/do_kiem/iters9_M_spring_bytetrack_buffer_18_minarea36/log"
    res = evaluate_dataset_many_folders(gt_root, pred_root, pred_format='bbox', iou_threshold=0.3, show_false_img = False)
    pass
