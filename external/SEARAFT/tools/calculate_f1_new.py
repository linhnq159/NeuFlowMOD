import os
import csv
import cv2
import xml.etree.ElementTree as ET
import numpy as np

# -------------------------
# IoU (xywh)
# -------------------------
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (w1 * h1) + (w2 * h2) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)

# -------------------------
# VOC xml -> bboxes(xywh), size, filename/path (for image)
# -------------------------
def parse_voc_xml(xml_path):
    """
    Returns:
      bboxes_xywh: [[x,y,w,h], ...]
      size: (width,height) or None
      img_filename: from <filename> if exists else None
      img_path_in_xml: from <path> if exists else None
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_filename = None
    img_path_in_xml = None
    fn_node = root.find("filename")
    if fn_node is not None and fn_node.text:
        img_filename = fn_node.text.strip()

    path_node = root.find("path")
    if path_node is not None and path_node.text:
        img_path_in_xml = path_node.text.strip()

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

        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)

        # giữ filter y như bạn (area >= 36 và w,h < 100)
        if (w * h) >= 36 and w < 100 and h < 100:
            bboxes.append([xmin, ymin, w, h])

    size = (width, height) if width and height else None
    
    return bboxes, size, img_filename, img_path_in_xml

# -------------------------
# Prediction loader (CSV flexible)
# Your file: line.split(",") and using parts[2..6]
# -> mình làm robust hơn: auto lấy numeric tokens
# Output bbox xywh
# -------------------------
def _split_line_flexible(line: str):
    line = line.strip()
    if not line:
        return []
    if "," in line:
        return [p.strip() for p in line.split(",") if p.strip() != ""]
    return line.split()

def load_pred_file_bbox(path, pred_bbox_format="xywh"):
    """
    pred_bbox_format:
      - "xywh": tokens là x, y, w, h
      - "xyxy": tokens là x1, y1, x2, y2  (sẽ convert về xywh)
    File của bạn hiện split bằng ',' và dùng parts[2..6] => giữ đúng behavior đó.
    """
    preds = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            # cần tối thiểu tới index 5
            if len(parts) < 6:
                continue
            try:
                a = float(parts[2]); b = float(parts[3]); c = float(parts[4]); d = float(parts[5])
                score = float(parts[6]) if len(parts) >= 7 else 1.0
            except:
                continue

            if pred_bbox_format == "xywh":
                x, y, w, h = a, b, c, d
                bbox = [x, y, max(0.0, w), max(0.0, h)]
            elif pred_bbox_format == "xyxy":
                x1, y1, x2, y2 = a, b, c, d
                bbox = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
            else:
                raise ValueError("pred_bbox_format must be 'xywh' or 'xyxy'")

            cx = bbox[0] + bbox[2]/2
            cy = bbox[1] + bbox[3]/2
            width = bbox[2]
            height = bbox[3]
            scale = 1.6
            bbox = [cx - scale * width/2, cy - scale * height/2, scale * width, scale * height]

            preds.append({'bbox': bbox, 'score': float(score)})
    return preds

# -------------------------
# Gather GT list
# -------------------------
def gather_dataset_pairs(gt_root):
    pairs = []
    for dirpath, _, filenames in os.walk(gt_root):
        for fname in filenames:
            if not fname.lower().endswith('.xml'):
                continue
            full = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full, gt_root)
            rel_noext = os.path.splitext(rel_path)[0]
            image_id_rel = rel_noext.replace(os.sep, '/')
            pairs.append((image_id_rel, full))
    return pairs

def find_pred_file_for(image_id_rel, gt_xml_path, pred_root):
    candidate1 = os.path.join(pred_root, image_id_rel.replace('annotations/', '').replace('/', os.sep) + '.txt')
    if os.path.isfile(candidate1):
        return candidate1

    basename = os.path.splitext(os.path.basename(image_id_rel))[0]
    candidate2 = os.path.join(pred_root, basename + '.txt')
    if os.path.isfile(candidate2):
        return candidate2

    candidate3 = os.path.splitext(gt_xml_path)[0] + '.txt'
    if os.path.isfile(candidate3):
        return candidate3

    return None

# -------------------------
# Image path resolver
# -------------------------
def find_image_for_xml(xml_path, image_root=None, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """
    Try to locate image for this xml:
      1) <path> in xml
      2) image_root/<filename> in xml
      3) same basename as xml in same folder (xml_dir/base + ext)
      4) image_root/<xml_basename + ext>
    """
    try:
        _, _, img_filename, img_path_in_xml = parse_voc_xml(xml_path)
    except:
        img_filename, img_path_in_xml = None, None
        
    xml_path_list = xml_path.split("/")
    img_path_in_xml = xml_path_list[-3] + "/images/" + img_filename if img_filename else None
    
    img_path_in_xml = os.path.join(image_root, img_path_in_xml) if image_root and img_path_in_xml else None

    if img_path_in_xml and os.path.isfile(img_path_in_xml):
        return img_path_in_xml

    if image_root and img_filename:
        cand = os.path.join(image_root, img_filename)
        if os.path.isfile(cand):
            return cand

    base = os.path.splitext(os.path.basename(xml_path))[0]
    xml_dir = os.path.dirname(xml_path)

    for e in exts:
        cand = os.path.join(xml_dir, base + e)
        if os.path.isfile(cand):
            return cand

    if image_root:
        for e in exts:
            cand = os.path.join(image_root, base + e)
            if os.path.isfile(cand):
                return cand

    return None

# -------------------------
# Draw helpers
# -------------------------
def _draw_xywh(img, box, color, text=None, thickness=2):
    x, y, w, h = box
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        cv2.putText(img, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# -------------------------
# Per-image match + visualize
# -------------------------
def match_image_gts_preds(gts_xywh, preds_xywh_scored, iou_threshold):
    """
    Returns:
      pred_is_tp: [bool]
      pred_best_iou: [float]
      pred_best_gt_idx: [int]  (-1 nếu không match được)
      gt_matched: [bool]
    """
    gt_matched = np.zeros(len(gts_xywh), dtype=bool)
    pred_is_tp = [False] * len(preds_xywh_scored)
    pred_best_iou = [0.0] * len(preds_xywh_scored)
    pred_best_gt_idx = [-1] * len(preds_xywh_scored)

    order = list(range(len(preds_xywh_scored)))
    order.sort(key=lambda i: preds_xywh_scored[i]['score'], reverse=True)

    for pi in order:
        pbox = preds_xywh_scored[pi]['bbox']
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gts_xywh):
            cur = iou(pbox, gt)
            if cur > best_iou:
                best_iou = cur
                best_gi = gi

        pred_best_iou[pi] = best_iou
        pred_best_gt_idx[pi] = best_gi

        if best_iou >= iou_threshold and best_gi != -1 and (not gt_matched[best_gi]):
            pred_is_tp[pi] = True
            gt_matched[best_gi] = True
        else:
            pred_is_tp[pi] = False

    return pred_is_tp, pred_best_iou, pred_best_gt_idx, gt_matched


# -------------------------
# Main evaluation: per-image CSV + visualize
# -------------------------
def evaluate_dataset_many_folders_with_vis(
    gt_root,
    pred_root,
    iou_threshold=0.3,
    score_threshold=0.0,
    image_root=None,                 # NEW: where images are
    export_csv_path="./per_image_report.csv",
    export_vis_dir="./vis_out",       # NEW: save annotated images
    verbose=True,
    visualize=True,
    export_det_csv_path=None,   # NEW
    log_det_csv=True            # NEW
):
    os.makedirs(os.path.dirname(export_csv_path) or ".", exist_ok=True)
    os.makedirs(export_vis_dir, exist_ok=True)

    pairs = gather_dataset_pairs(gt_root)
    if not pairs:
        raise RuntimeError("No GT xml files found under gt_root: " + str(gt_root))

    rows = []
    
    det_writer = None
    det_f = None
    if log_det_csv:
        if export_det_csv_path is None:
            export_det_csv_path = os.path.splitext(export_csv_path)[0] + "_per_det.csv"
        os.makedirs(os.path.dirname(export_det_csv_path) or ".", exist_ok=True)
        det_f = open(export_det_csv_path, "w", newline="")
        det_writer = csv.writer(det_f)
        det_writer.writerow([
            "image_id",
            "type",                 # PRED hoặc FN
            "pred_idx",
            "pred_score",
            "pred_x", "pred_y", "pred_w", "pred_h",
            "pred_label",           # TP/FP
            "best_gt_idx",
            "best_iou",
            "gt_x", "gt_y", "gt_w", "gt_h",
        ])
    
    total_TP = total_FP = total_FN = total_GT = total_pred = 0
    matched_ious_all = []

    for image_id_rel, xml_path in pairs:
        gts, size, _, _ = parse_voc_xml(xml_path)

        pred_file = find_pred_file_for(image_id_rel, xml_path, pred_root)
        preds = []
        if pred_file is not None:
            preds = load_pred_file_bbox(pred_file, pred_bbox_format="xywh")   # hoặc "xyxy"
            preds = [p for p in preds if p['score'] >= score_threshold]

        pred_is_tp, pred_best_iou, pred_best_gt_idx, gt_matched = match_image_gts_preds(gts, preds, iou_threshold)


        TP = int(sum(1 for v in pred_is_tp if v))
        FP = int(sum(1 for v in pred_is_tp if not v))
        FN = int(len(gts) - int(np.sum(gt_matched)))
        GT = int(len(gts))
        PRED = int(len(preds))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / GT if GT > 0 else 0.0
        mean_iou_tp = float(np.mean([pred_best_iou[i] for i in range(len(preds)) if pred_is_tp[i]])) if TP > 0 else 0.0

        rows.append([image_id_rel, GT, PRED, TP, FP, FN, precision, recall, mean_iou_tp])

        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_GT += GT
        total_pred += PRED
        if TP > 0:
            matched_ious_all.extend([pred_best_iou[i] for i in range(len(preds)) if pred_is_tp[i]])

        # --------- DETAILED LOG PER PRED ---------
        if det_writer is not None:
            # log each prediction
            for pi, p in enumerate(preds):
                pbox = p["bbox"]
                sc = p["score"]
                best_gi = pred_best_gt_idx[pi]
                biou = pred_best_iou[pi]
                label = "TP" if pred_is_tp[pi] else "FP"

                gt_box = gts[best_gi] if (best_gi is not None and best_gi >= 0 and best_gi < len(gts)) else None
                if gt_box is None:
                    gx = gy = gw = gh = ""
                else:
                    gx, gy, gw, gh = gt_box

                det_writer.writerow([
                    image_id_rel,
                    "PRED",
                    pi,
                    f"{sc:.6f}",
                    f"{pbox[0]:.2f}", f"{pbox[1]:.2f}", f"{pbox[2]:.2f}", f"{pbox[3]:.2f}",
                    label,
                    best_gi,
                    f"{biou:.6f}",
                    gx if gx == "" else f"{gx:.2f}",
                    gy if gy == "" else f"{gy:.2f}",
                    gw if gw == "" else f"{gw:.2f}",
                    gh if gh == "" else f"{gh:.2f}",
                ])

            # log FN GTs (those not matched)
            for gi, gt_box in enumerate(gts):
                if not gt_matched[gi]:
                    gx, gy, gw, gh = gt_box
                    det_writer.writerow([
                        image_id_rel,
                        "FN",
                        "", "",
                        "", "", "", "",
                        "FN",
                        gi,
                        f"{0.0:.6f}",
                        f"{gx:.2f}", f"{gy:.2f}", f"{gw:.2f}", f"{gh:.2f}",
                    ])

        if visualize:
            # --------- VISUALIZE ---------
            img_path = find_image_for_xml(xml_path, image_root=image_root)
            # print("xml_path : ", xml_path)
            # print("img_path : ", img_path)
            if img_path is not None and os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    # Draw GT: FN đỏ, matched GT xanh lá
                    for gi, gt_box in enumerate(gts):
                        if gt_matched[gi]:
                            _draw_xywh(img, gt_box, (0, 255, 0), text="GT-M")   # green
                        else:
                            _draw_xywh(img, gt_box, (0, 0, 255), text="FN")     # red

                    # Draw preds: TP xanh lá + score/iou, FP xanh dương
                    for pi, p in enumerate(preds):
                        box = p['bbox']
                        sc = p['score']
                        biou = pred_best_iou[pi]
                        if pred_is_tp[pi]:
                            _draw_xywh(img, box, (0, 255, 0), text=f"TP s={sc:.2f} iou={biou:.2f}", thickness=2)
                        else:
                            _draw_xywh(img, box, (255, 0, 0), text=f"FP s={sc:.2f} iou={biou:.2f}", thickness=2)

                    # header text
                    cv2.putText(img, f"{image_id_rel} | GT={GT} PRED={PRED} TP={TP} FP={FP} FN={FN}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    # save
                    # out_name = image_id_rel.replace("/", "__") + ".jpg"
                    out_name = image_id_rel.split('/')[-1] + ".jpg"
                    name_video = image_id_rel.split('/')[0]
                    output_path = os.path.join(export_vis_dir, name_video)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    out_path = os.path.join(output_path, out_name)
                    cv2.imwrite(out_path, img)

    # write CSV
    with open(export_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "gt", "pred", "tp", "fp", "fn", "precision", "recall", "mean_iou_tp"])
        w.writerows(rows)

    # summary
    dataset_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    dataset_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    dataset_f1 = (2 * dataset_precision * dataset_recall / (dataset_precision + dataset_recall)) if (dataset_precision + dataset_recall) > 0 else 0.0
    dataset_mean_iou = float(np.mean(matched_ious_all)) if matched_ious_all else 0.0

    if verbose:
        print("=== Per-image CSV + Visualization ===")
        print(f"Images scanned     : {len(pairs)}")
        print(f"Total GT / Pred    : {total_GT} / {total_pred}")
        print(f"TP / FP / FN       : {total_TP} / {total_FP} / {total_FN}")
        print(f"Precision / Recall : {dataset_precision:.4f} / {dataset_recall:.4f}")
        print(f"F1                 : {dataset_f1:.4f}")
        print(f"Mean IoU (TP)      : {dataset_mean_iou:.4f}")
        print(f"CSV saved to       : {export_csv_path}")
        print(f"Vis images saved to: {export_vis_dir}")
        print(f"Detail CSV saved to : {export_det_csv_path}")
        print("=====================================")

    if det_f is not None:
       det_f.close()

    return {
        "total_images": len(pairs),
        "total_gt": total_GT,
        "total_pred": total_pred,
        "TP": total_TP,
        "FP": total_FP,
        "FN": total_FN,
        "precision": float(dataset_precision),
        "recall": float(dataset_recall),
        "f1": float(dataset_f1),
        "mean_iou_tp": float(dataset_mean_iou),
        "csv": export_csv_path,
        "vis_dir": export_vis_dir
    }

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    gt_root = "/data/linhnq11/Data/MOD/PESMOD"
    pred_root = "/data/linhnq11/MOD/SEA-RAFT/output/do_kiem/input_optical_flow2segment/log"
    image_root = "/data/linhnq11/Data/MOD/PESMOD"
    
    export_csv_path = pred_root + ".csv"
    export_vis_dir = pred_root[:-3] + "vis_05"
    os.makedirs(export_vis_dir, exist_ok=True)

    res = evaluate_dataset_many_folders_with_vis(
        gt_root=gt_root,
        pred_root=pred_root,
        iou_threshold=0.25,
        score_threshold=0.0,
        image_root=image_root,
        export_csv_path=export_csv_path,
        export_vis_dir=export_vis_dir,
        verbose=True,
        visualize=True,
        export_det_csv_path=None,   # NEW
        log_det_csv=True            # NEW
    )
