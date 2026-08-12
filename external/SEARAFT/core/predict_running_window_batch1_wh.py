from typing import Callable, List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
import os, shutil, glob
from tqdm import tqdm
import torch
from torchvision.transforms import Resize
from utils.general import non_max_suppression
from sahi1.postprocess.combine import (
    GreedyNMMPostprocess, 
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
)
import math

POSTPROCESS_NAME_TO_CLASS = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "LSNMS": LSNMSPostprocess,
}

def recreate_dir(dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

class DetectionROI:
    def __init__(self, roi_shape = 640, tile_size = 64, max_age = 30, dtype = torch.float16, deblur = False):
        """
        roi_shape: either scalar (old behavior, square) or tuple (roi_h, roi_w)
        tile_size: tile size (tiles are square)
        """
        self.tile_size = tile_size 
        # self.roi_shape = roi_shape
        # self.num_cols = roi_shape//tile_size
        
        # support scalar or tuple
        if isinstance(roi_shape, (list, tuple, np.ndarray)):
            self.roi_h = int(roi_shape[0])
            self.roi_w = int(roi_shape[1])
        else:
            self.roi_h = int(roi_shape)
            self.roi_w = int(roi_shape)
            
        self.num_cols = self.roi_w // tile_size
        self.num_rows = self.roi_h // tile_size
        
        
        
        self.max_age = max_age
        blank_ROI = torch.zeros((3, self.roi_h, self.roi_w), dtype=dtype)
        self.init_ROI(blank_ROI, False)
        self.frame_id = 0
        self.psf_size = 3
        # self.kernel = np.ones((self.psf_size, self.psf_size))/(self.psf_size**2)
        self.kernel = torch.ones((self.psf_size, self.psf_size))/(self.psf_size**2)
        self.K = 0.01
        self.dtype = dtype
        self.deblur = False
    
    def crop_im(self, im, box):
        x1n, y1n, x2n, y2n = box
        x1n = int(x1n)
        y1n = int(y1n)
        x2n = int(x2n)
        y2n = int(y2n)
        # im expected shape: C x H x W
        im_part = im[:, y1n : y2n, x1n: x2n]
        im_part = Resize((self.tile_size, self.tile_size))(im_part)
        return im_part
                                              
    def paste_to_tile(self, im2paste, world_pos, tile_id):
        ROI_x1, ROI_y1, ROI_x2, ROI_y2 = self.tiles[tile_id]['ROI_pos']
        self.tiles[tile_id]['world_pos'] = world_pos
        self.tiles[tile_id]['available'] = False
        self.ROI[:, ROI_y1 : ROI_y2, ROI_x1 : ROI_x2] = im2paste
    
    def init_ROI(self, blankROI, include_left):
        # blankROI should be C x H x W
        if blankROI.shape[1] != self.roi_h or blankROI.shape[2] != self.roi_w:
            self.ROI = Resize((self.roi_h, self.roi_w))(blankROI)
        else:
            self.ROI = blankROI

        self.tiles, self.tmp_tiles = {}, []

        # --- Compatibility mode: if ROI is square, preserve original behavior exactly ---
        if self.roi_h == self.roi_w:
            # original used num_cols and a single loop range(self.num_cols)
            if include_left:
                j = 0
            else:
                j = max(0, self.num_cols - 1)
            for i in range(self.num_cols):
                self.tiles.update({
                    i: {
                        'ROI_pos': [j*self.tile_size,
                                    i*self.tile_size,
                                    (j+1)*self.tile_size,
                                    (i+1)*self.tile_size],
                        'world_pos': [],
                        'available': True,
                        'last_frame': 0,
                        'obj_list': [],
                        'is_deblur': False
                    }
                })
        else:
            # New rectangular behavior: place a vertical column of tiles in either left or right column index
            if include_left:
                col_idx = 0
            else:
                col_idx = max(0, self.num_cols - 1)
            for row in range(self.num_rows):
                x1 = col_idx * self.tile_size
                y1 = row * self.tile_size
                x2 = x1 + self.tile_size
                y2 = y1 + self.tile_size
                self.tiles.update({
                    row: {
                        'ROI_pos': [x1, y1, x2, y2],
                        'world_pos': [],
                        'available': True,
                        'last_frame': 0,
                        'obj_list': [],
                        'is_deblur': False
                    }
                })

        self.isBlank = True
        # ROI_region stored as [[tl_x, tl_y], [br_x, br_y]]
        self.ROI_region = [[self.roi_w, self.roi_h], [0, 0]]

    def update_current_view(self, frame):
        """
        Update ROI from tmp_tiles and paste crops into corresponding ROI tiles.
        - If ROI is square: map tmp_tiles[idx] -> tiles[idx] (exact old behavior).
        - Else: map using sorted tile keys to avoid KeyError.
        """
        if len(self.tiles) == 0:
            return

        tile_keys = sorted(self.tiles.keys())  # stable ordering
        n_tiles = len(tile_keys)

        for idx, tmp_tile in enumerate(self.tmp_tiles):
            # Compatibility: if roi square, original expected tile_id == idx
            if self.roi_h == self.roi_w:
                tile_id = idx
            else:
                tile_id = tile_keys[idx] if idx < n_tiles else tile_keys[idx % n_tiles]

            # safety: if tile_id not present, skip (avoid KeyError)
            if tile_id not in self.tiles:
                # (Option) you can `break` here if you prefer to stop when out of tiles
                continue

            # update base tile
            self.tiles[tile_id]['last_frame'] = tmp_tile.get('last_frame', 0)
            self.tiles[tile_id]['world_pos'] = tmp_tile['world_pos']
            self.tiles[tile_id]['available'] = False

            # crop from full frame using world_pos and paste into tile
            try:
                im_part = self.crop_im(frame, tmp_tile['world_pos'])
                self.paste_to_tile(im_part, tmp_tile['world_pos'], tile_id)
            except Exception as e:
                print(f"Warning: failed to crop/paste tmp_tile idx={idx}, tile_id={tile_id}: {e}")
                continue

            # handle deblur tile if enabled and a corresponding deblur tile exists
            if self.deblur and len(self.tmp_tiles) < 13:
                deblur_tid_candidate = tile_id + 13
                if deblur_tid_candidate in self.tiles:
                    deblur_tid = deblur_tid_candidate
                    self.tiles[deblur_tid]['last_frame'] = tmp_tile.get('last_frame', 0)
                    self.tiles[deblur_tid]['world_pos'] = tmp_tile['world_pos']
                    self.tiles[deblur_tid]['available'] = False
                    self.tiles[deblur_tid]['is_deblur'] = True
                    try:
                        deblured_im = self.apply_deblur(im_part)
                        self.paste_to_tile(deblured_im, tmp_tile['world_pos'], deblur_tid)
                    except Exception as e:
                        print(f"Warning: failed deblur for idx={idx}, deblur_tid={deblur_tid}: {e}")
                        # continue without crashing

        if len(self.tmp_tiles) > 0:
            # match original ROI_region assignment when square
            if self.roi_h == self.roi_w:
                first_tile_id = 0
                last_tile_id = min(len(self.tmp_tiles) - 1, max(self.tiles.keys()))
            else:
                first_tile_id = tile_keys[0]
                last_tile_id = tile_keys[min(len(self.tmp_tiles) - 1, n_tiles - 1)]

            self.ROI_region[0] = [self.tiles[first_tile_id]['ROI_pos'][0], self.tiles[first_tile_id]['ROI_pos'][1]]
            self.ROI_region[1] = [self.tiles[last_tile_id]['ROI_pos'][2], self.tiles[last_tile_id]['ROI_pos'][3]]

    def update_current_view_with_lb(self, frame):
        norm_boxes = []
        for tile_id, tmp_tile in enumerate(self.tmp_tiles):
            self.tiles[tile_id]['world_pos'] = tmp_tile['world_pos']
            self.tiles[tile_id]['available'] = False

            world_pos = tmp_tile['world_pos']
            ROI_pos = self.tiles[tile_id]['ROI_pos']
            
            for obj in tmp_tile.get('obj_list', []):
                cls_id = obj[0]
                # For x axis use roi_w, for y axis use roi_h when normalizing
                x1 = ((obj[1] - world_pos[0]) * self.tile_size / (world_pos[2] - world_pos[0]) + ROI_pos[0]) / self.roi_w
                y1 = ((obj[2] - world_pos[1]) * self.tile_size / (world_pos[3] - world_pos[1]) + ROI_pos[1]) / self.roi_h
                x2 = ((obj[3] - world_pos[0]) * self.tile_size / (world_pos[2] - world_pos[0]) + ROI_pos[0]) / self.roi_w
                y2 = ((obj[4] - world_pos[1]) * self.tile_size / (world_pos[3] - world_pos[1]) + ROI_pos[1]) / self.roi_h
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + x2)/2
                cy = (y1 + y2)/2
                norm_boxes.append([cls_id, cx, cy, w, h])

            im_part = self.crop_im(frame, tmp_tile['world_pos'])                
            self.paste_to_tile(im_part, tmp_tile['world_pos'], tile_id)
        return norm_boxes

    def print_tiles(self):
        print("Tile list: ")
        for tid, tile in self.tiles.items():
            if not tile['available']:
                print(f"\n {tid} - world pos:{tile['world_pos']} - last frame id: {tile['last_frame']} ")

    @property
    def image(self):
        return self.ROI
      
    def viz(self, frame, roi_pred = [], frame_pred = []):
        viz = self.ROI.detach().clone().numpy()
        viz = viz.transpose(1,2,0).copy()
        viz = np.array(viz*255).astype(np.uint8)
        
        for pred in roi_pred:
            x1, y1, x2, y2 = pred[:4]
            viz = cv2.rectangle(viz, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

        for tile_id, tile in self.tiles.items():
            world_pos = tile['world_pos']
            ROI_x1, ROI_y1, ROI_x2, ROI_y2 = tile['ROI_pos']
            if len(world_pos):
                for pred in frame_pred:
                    cx, cy, w, h = (pred[0] + pred[2])/2, (pred[1] + pred[3])/2, pred[2] - pred[0], pred[3] - pred[1]
                    if pred[0] >= world_pos[0] and pred[1] >= world_pos[1] and pred[2] <= world_pos[2] and pred[3] <= world_pos[3]:
                        cx = (cx - world_pos[0])/(world_pos[2] - world_pos[0])
                        cy = (cy - world_pos[1])/(world_pos[3] - world_pos[1])
                        w = w * self.tile_size / (world_pos[2] - world_pos[0])
                        h = h * self.tile_size / (world_pos[3] - world_pos[1])
                        x1r = cx * self.tile_size + ROI_x1 - 0.5*w
                        y1r = cy * self.tile_size + ROI_y1 - 0.5*h
                        x2r = x1r + w
                        y2r = y1r + h

                        viz = cv2.rectangle(viz, (int(x1r), int(y1r)), (int(x2r), int(y2r)), (255,0,0), 2)

        return viz
            
    def align_box(self, cx, cy, size, W, H):
        x1, y1 = max(0, cx - size/2), max(0, cy - size/2)
        x2, y2 = min(x1 + size, W), min(y1 + size, H)
        x1, y1 = x2 - size, y2 - size
        return x1, y1, x2, y2
    
    def update_by_boxlist(self, img, boxes, frame_id, blankROI, include_left):
        self.init_ROI(blankROI, include_left)

        self.frame_id = frame_id
        H, W = img.shape[1], img.shape[2]

        if len(boxes) > 0:
            _boxes = []
            for tlbr in boxes:
                _boxes.append(W*H - (tlbr[3] - tlbr[1])*(tlbr[4] - tlbr[2])) 
            
            boxes = np.array(boxes)
            
            sorted_ids = np.argsort(_boxes, axis = 0)
            sorted_ids = sorted_ids.tolist()
            
            while len(sorted_ids) > 0:
                id = sorted_ids[0]

                _, x11, y11, x12, y12 = boxes[id]

                w1 = x12 - x11
                h1 = y12 - y11

                if max(w1, h1) <= 64:
                    wn = self.tile_size
                elif max(w1, h1) <= min(W, H)//2:
                    wn = 2*max(w1, h1)
                else:
                    wn = min(W, H)

                sorted_boxes = boxes[sorted_ids]
                
                near_box_indices = np.array(sorted_ids)[np.maximum(0, (np.minimum(sorted_boxes[:, 3], x11 + wn) - np.maximum(sorted_boxes[:, 1], x12 - wn)))*np.maximum(0, 
                                                                       (np.minimum(sorted_boxes[:, 4], y11 + wn) - np.maximum(sorted_boxes[:, 2], y12 - wn))) / (
                                                                           (sorted_boxes[:, 3] - sorted_boxes[:, 1])*(sorted_boxes[:, 4] - sorted_boxes[:, 2])) >= 0.9
                                                        ]

                if len(near_box_indices) == 0:
                    sorted_ids.remove(id)
                    continue

                t_cx, t_cy = (x11 + x12)/2, (y11 + y12)/2
                last_frames, track_ids = [], []
                scores = []
                obj_list = []
                for j in near_box_indices:
                    cls_id, x1, y1, x2, y2 = boxes[j]
                    
                    # Update tile's center t_cx, t_cy to capture current box
                    t_cxn, t_cyn = (t_cx + (x2 + x1)/2)/2, (t_cy + (y2 + y1)/2)/2
                    t_x1n, t_y1n, t_x2n, t_y2n = self.align_box(t_cxn, t_cyn, wn, W, H)
                    
                    if int(t_x1n > x11) + int(t_y1n > y11) + int(t_x2n < x12) + int(t_y2n < y12) > 0:
                        continue
                    else:
                        t_cx, t_cy = (t_x1n + t_x2n)/2, (t_y1n + t_y2n)/2
                        sorted_ids.remove(j)
                        obj_list.append([cls_id, x1, y1, x2, y2])
                   
                t_x1, t_y1, t_x2, t_y2 = self.align_box(t_cx, t_cy, wn, W, H)

                # Update tile's information
                self.tmp_tiles.append({
                                        'obj_list': obj_list,
                                        'world_pos' : [t_x1, t_y1, t_x2, t_y2] ,
                                        })
            
        
        return self.update_current_view_with_lb(img)
            

    def update(self, img, tracker, frame_id, blankROI, include_left):
        self.init_ROI(blankROI, include_left)

        self.frame_id = frame_id
        H, W = img.shape[1], img.shape[2]

        if len(tracker.tracked_stracks) > 0 or len(tracker.lost_stracks) > 0:
            boxes = []
            list_tracks = tracker.tracked_stracks + tracker.lost_stracks
            _boxes = []
            print("\nNumber of tracks: ", len(list_tracks))
            for track in list_tracks:
                tlwh = track.tlwh
                tlbr = track.tlwh_to_tlbr(tlwh)
                if tlbr[0] > W or tlbr[1]  > H or tlbr[2] < 0 or tlbr[3] < 0: 
                    continue

                boxes.append([tlbr[0], tlbr[1], tlbr[2], tlbr[3]])
                print(len(boxes), ":", tlbr[0], ",", tlbr[1], ", ", tlbr[2] , ", ", tlbr[3])
                _boxes.append(W*H - (tlbr[2] - tlbr[0])*(tlbr[3] - tlbr[1])) 
            
            if len(boxes) == 0:
                boxes = np.zeros((0,4))
            else:
                boxes = np.array(boxes)
                boxes[:, 0] = np.clip(boxes[:, 0], 0, W)
                boxes[:, 1] = np.clip(boxes[:, 1], 0, H)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, W)
                boxes[:, 3] = np.clip(boxes[:, 3], 0, H)
            
            sorted_ids = np.argsort(_boxes, axis = 0)
            sorted_ids = sorted_ids.tolist()
            
            while len(sorted_ids) > 0:
                id = sorted_ids[0]

                x11, y11, x12, y12 = boxes[id]

                w1 = x12 - x11
                h1 = y12 - y11

                if max(w1, h1) <= 64:
                    wn = self.tile_size
                elif max(w1, h1) <= min(W, H)//2:
                    wn = 2*max(w1, h1)
                else:
                    wn = min(W, H)

                sorted_boxes = boxes[sorted_ids]
                
                near_box_indices = np.array(sorted_ids)[np.maximum(0, (np.minimum(sorted_boxes[:, 2], x11 + wn) - np.maximum(sorted_boxes[:, 0], x12 - wn)))*np.maximum(0, 
                                                                       (np.minimum(sorted_boxes[:, 3], y11 + wn) - np.maximum(sorted_boxes[:, 1], y12 - wn))) / (
                                                                           (sorted_boxes[:, 2] - sorted_boxes[:, 0])*(sorted_boxes[:, 3] - sorted_boxes[:, 1])) >= 0.9
                                                        ]
                
                if len(near_box_indices) == 0:
                    sorted_ids.remove(id)
                    continue

                t_cx, t_cy = (x11 + x12)/2, (y11 + y12)/2
                last_frames, track_ids = [], []
                scores = []

                for j in near_box_indices:
                    x1, y1, x2, y2 = boxes[j]
                    
                    # Update tile's center t_cx, t_cy to capture current box
                    t_cxn, t_cyn = (t_cx + (x2 + x1)/2)/2, (t_cy + (y2 + y1)/2)/2
                    t_x1n, t_y1n, t_x2n, t_y2n = self.align_box(t_cxn, t_cyn, wn, W, H)
                    
                    if int(t_x1n > x11) + int(t_y1n > y11) + int(t_x2n < x12) + int(t_y2n < y12) > 0:
                        continue
                    else:
                        t_cx, t_cy = (t_x1n + t_x2n)/2, (t_y1n + t_y2n)/2
                        last_frames.append(list_tracks[j].end_frame)
                        track_ids.append(list_tracks[j].track_id)
                        scores.append(
                                        (1 - (frame_id - list_tracks[j].end_frame)/tracker.max_time_lost) + (list_tracks[j].tracklet_len/frame_id)
                                    )
                        sorted_ids.remove(j)
                   
                t_x1, t_y1, t_x2, t_y2 = self.align_box(t_cx, t_cy, wn, W, H)
                if len(scores) == 0:
                    assert len(last_frames) == len(scores), "Logical bug"
                    sorted_ids.remove(id)
                    continue
                else:
                    self.tmp_tiles.append({
                                            'world_pos' : [t_x1, t_y1, t_x2, t_y2] ,
                                            'score' : max(scores),
                                            'last_frame' : max(last_frames),
                                            })
            
            if len(self.tmp_tiles) > 0:
                self.tmp_tiles = sorted(self.tmp_tiles, key = lambda x: x['score'], reverse=True)
        if len(self.tmp_tiles) > 0: 
            self.isBlank = False
        self.update_current_view(img)
            
        self.print_tiles()
        
    def wiener_filter(self, im, kernel, K):
        im = im.to(torch.float32)
        kernel = torch.ones((self.psf_size, self.psf_size))/(self.psf_size**2)
        
        # Perform FFT on the image and the kernel
        H = torch.fft.fft2(kernel, s = im.shape)
        G = torch.fft.fft2(im)
        
        # Weiner filter in frequency domain
        H_conj = torch.conj(H)
        H_power = torch.abs(H)**2
        Wiener_filter = H_conj/(H_power + K)
        
        F_hat = G*Wiener_filter
        f_hat = torch.abs(torch.fft.ifft2(F_hat))
        
        return torch.clip(f_hat, 0, 255).to(torch.half)
    
    def apply_deblur(self, im: torch.Tensor):
        channels = [(im[i]*255).to(torch.uint8) for i in range(im.shape[0])]
        deblurred_channels = [self.wiener_filter(channel, self.kernel, self.K) for channel in channels]
        res = torch.stack(deblurred_channels)
        res = res.to(self.ROI.dtype)  # uint8 to fp16/32 
        res /= 255.0  # 0 - 255 to 0.0 - 1.0
        return res

class Running_window_prediction:
    def __init__(self, 
                 detection_model: Callable, 
                 slice_size: Tuple, 
                 overlap: Tuple, 
                 max_age = 30,
                 tile_size = 64,
                 model_confidence_threshold = 0.3,
                 nms_thresh = 0.45,
                 nms_classes = 1,
                 agnostic_nms = False,
                 ignored_classes = [1],
                 apply_deblur = False): 
        
        self.slice_height = slice_size[0]
        self.slice_width = slice_size[1]
        self.overlap_height_ratio = overlap[0]
        self.overlap_width_ratio = overlap[1]
        self.detector = detection_model
        self.inference_time = 0
        self.post_process_time = 0
        self.slicing_time = 0
        self.conf_thres = model_confidence_threshold
        self.nms_thresh = nms_thresh
        self.nms_classes = nms_classes
        self.agnostic_nms = agnostic_nms
        self.frame_id = 0
        self.ignored_classes = ignored_classes
        
        p = next(self.detector.parameters())
        self.device = p.device
        self.dtype = p.dtype
        # pass both h and w to DetectionROI so it is rectangular if needed
        self.ROI = DetectionROI((self.slice_height, self.slice_width), tile_size = tile_size, max_age = max_age, dtype = self.dtype, deblur = apply_deblur)
        self.prediction_results = None
        self.out_frame = None
        
    def simple_slicing(self, H, W):
        mesh_x1 = np.arange(0, W, self.slice_width*(1-self.overlap_width_ratio)).astype(np.int32)
        mesh_y1 = np.arange(0, H, self.slice_height*(1-self.overlap_height_ratio)).astype(np.int32)
        
        mesh_x2 = np.clip(mesh_x1 + self.slice_width, 0, W)
        mesh_y2 = np.clip(mesh_y1 + self.slice_height, 0, H)
        
        mesh_x1 = np.unique(np.minimum(mesh_x2 - self.slice_width, mesh_x1))
        mesh_y1 = np.unique(np.minimum(mesh_y2 - self.slice_height, mesh_y1))
        mesh_x, mesh_y = np.meshgrid(mesh_x1, mesh_y1)
        
        self.slices = np.stack([mesh_x.flatten(), mesh_y.flatten()], axis = -1).tolist()
        self.num_slices = len(self.slices)
        
    def update_running_id(self):
        self.running_id = (self.running_id - 1)%self.num_slices
    
    def inference_frame(self, frame, tracker, frame_id, save_dir, name): 
        self.frame_id = frame_id
        self.out_frame = frame.permute(1,2,0).detach().numpy()*255
        self.out_frame = np.ascontiguousarray( self.out_frame, dtype=np.uint8)
        
        roi = Image.fromarray(self.ROI.viz(self.out_frame.copy(), [], []))
        roi.save(os.path.join(save_dir / 'roi_org', "{}_{}.jpg".format(name, frame_id+10000)))
        self.roi_preds, self.frame_preds = [], []
        
        if frame_id == 1:
            self.simple_slicing(frame.shape[1], frame.shape[2])
            self.running_id = self.num_slices - 1
            self.another_running_id = -1
            
        if self.running_id < 0:
            self.running_id = self.num_slices-1

        shift_amount1 = self.slices[self.running_id]
        slice1 = frame[:, shift_amount1[1] : shift_amount1[1] + self.slice_height, shift_amount1[0] : shift_amount1[0] + self.slice_width]

        include_left = (self.running_id == self.num_slices - 1) or (self.running_id == self.num_slices//2 - 1)
        self.ROI.update(frame, tracker, frame_id, slice1, include_left)

        roi_im = self.ROI.image
        
        print("slice1 shape : ", slice1.shape)
        print("roi_im shape : ", roi_im.shape)
        
        self.prediction_results = self.get_prediction(  [slice1, roi_im], 
                                                        [shift_amount1, [-1,-1]])

        self.update_running_id()

        roi = Image.fromarray(self.ROI.viz(self.out_frame, self.roi_preds, self.frame_preds))
        roi.save(os.path.join(save_dir / 'roi_pd', "{}_{}.jpg".format(name, frame_id+10000)))
        
        
    def get_prediction(self, ims, shift_amount = [0,0]):
        input_batch = ims[0].unsqueeze(0)
        # input_batch = torch.stack(ims,dim = 0)
        print("ims[0] shape: ", ims[0].shape)
        print("Input batch orin shape: ", input_batch.shape)
        
        prediction_result = self.detector(input_batch.to(self.device))[0]
        prediction_result = non_max_suppression(prediction_result, 
                                                self.conf_thres, 
                                                self.nms_thresh, 
                                                self.nms_classes, 
                                                self.agnostic_nms)
    
        boxes_before_nms = []
        print("Inserted region: ", self.ROI.ROI_region)
        ROI_tl = self.ROI.ROI_region[0]
        ROI_br = self.ROI.ROI_region[1]
        for pred_per_slice, shift_amount_per_slice in zip(prediction_result, shift_amount):
            pred_per_slice = pred_per_slice.cpu().detach().numpy()
            boxes = pred_per_slice[:, :4]
            boxes[:, 0] = np.clip(boxes[:, 0], 0, self.slice_width)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, self.slice_width)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, self.slice_height)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, self.slice_height)
            scores = pred_per_slice[:, 4]
            cls_ids = pred_per_slice[:, 5]
            for box, score, cls_id in zip(boxes, scores, cls_ids):
                cls_id = int(cls_id)
                if cls_id in self.ignored_classes:
                    continue
                x1, y1, x2, y2 = box
                self.roi_preds.append([x1, y1, x2, y2, score, cls_id])
                # if the predicted box entirely inside ROI region (in ROI coordinates)
                if x1 >= ROI_tl[0] and y1 >= ROI_tl[1] and x2 <= ROI_br[0] and y2 <= ROI_br[1]:
                    if (math.floor(x2/self.ROI.tile_size) - math.floor(x1 / self.ROI.tile_size)) + (math.floor(y2 / self.ROI.tile_size) - math.floor(y1 / self.ROI.tile_size)) > 0:
                        continue
                    tile_id = math.floor(y1 / self.ROI.tile_size)

                    roi_x1, roi_y1, _, _ = self.ROI.tiles[tile_id]['ROI_pos']
                    x1r = (x1 - roi_x1)/self.ROI.tile_size
                    y1r = (y1 - roi_y1)/self.ROI.tile_size
                    x2r = (x2 - roi_x1)/self.ROI.tile_size
                    y2r = (y2 - roi_y1)/self.ROI.tile_size
                    
                    world_x1, world_y1, world_x2, world_y2 = self.ROI.tiles[tile_id]['world_pos']
                    x1 = world_x1 + x1r*(world_x2 - world_x1)
                    x2 = world_x1 + x2r*(world_x2 - world_x1)
                    y1 = world_y1 + y1r*(world_y2 - world_y1)
                    y2 = world_y1 + y2r*(world_y2 - world_y1)
                else:
                    # map slice-local coords to frame coords using shift amount
                    x1 = x1 + shift_amount_per_slice[0]
                    x2 = x2 + shift_amount_per_slice[0]
                    y1 = y1 + shift_amount_per_slice[1]
                    y2 = y2 + shift_amount_per_slice[1]

                boxes_before_nms.append([x1,y1,x2,y2, score, cls_id])
        
        object_prediction_list = []
        if len(boxes_before_nms):
            object_prediction_list = non_largest_suppression(boxes_before_nms, 0.5)

        self.roi_preds = non_largest_suppression(self.roi_preds, 0.5)
        self.frame_preds = non_largest_suppression(self.frame_preds, 0.5)
        return np.array(object_prediction_list)


def non_largest_suppression(predictions, iou_thres):
    predictions = sorted(predictions, key = lambda x: x[4]*(x[2] - x[0])*(x[3] - x[1]), reverse=True)
    results = []
    
    while len(predictions):
        merged_ids = [0]
        merge_x1, merge_y1, merge_x2, merge_y2, merge_score, merge_id = predictions[0]

        for j in range(1, len(predictions)):
            if j in merged_ids or predictions[j][-1] != merge_id:
                continue
            
            ia = max(0, min(merge_x2, predictions[j][2]) - max(merge_x1, predictions[j][0])) * max(0, min(merge_y2, predictions[j][3]) - max(merge_y1, predictions[j][1]))
            
            sa = min(   (merge_x2 - merge_x1)*(merge_y2 - merge_y1),
                        (predictions[j][2] - predictions[j][0])*(predictions[j][3] - predictions[j][1]) )
            if ia/sa >= iou_thres:
                merged_ids.append(j)
                merge_x1 = min(merge_x1, predictions[j][0]) 
                merge_y1 = min(merge_y1, predictions[j][1]) 
                merge_x2 = max(merge_x2, predictions[j][2]) 
                merge_y2 = max(merge_y2, predictions[j][3]) 
        
        merged_ids = sorted(merged_ids, reverse = True)     
        for merged_id in merged_ids:
            predictions.remove(predictions[merged_id])    
        results.append([merge_x1, merge_y1, merge_x2, merge_y2, merge_score, merge_id])

    return results
