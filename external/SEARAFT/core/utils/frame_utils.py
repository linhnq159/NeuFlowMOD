import numpy as np
from PIL import Image
from os.path import *
import re
import h5py

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])

def readFlo5Flow(filename):
    with h5py.File(filename, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?")
        return f["flow"][()]

def writeFlo5File(flow, filename):
    with h5py.File(filename, "w") as f:
        f.create_dataset("flow", data=flow, compression="gzip", compression_opts=5)

def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    elif ext == '.flo5':
        return readFlo5Flow(file_name)
    return []

def filter_flow(flow, T_mag=None, min_area=0, max_area=10000):
    """
    Lọc optical flow dựa trên ngưỡng độ lớn (magnitude) và diện tích vùng chuyển động.

    Args:
        flow (np.ndarray): Optical flow, dạng [H,W,2]
        T_mag (float, optional): Ngưỡng độ lớn (px/frame). Những pixel có magnitude < T_mag sẽ bị đặt 0. 
                                 Nếu None -> không dùng ngưỡng. Defaults to None.
        min_area (int, optional): Diện tích tối thiểu để giữ lại (tính theo pixel). 
                                  Vùng nhỏ hơn sẽ bị loại. Defaults to 0.

    Returns:
        np.ndarray: Flow đã được lọc, cùng shape [H,W,2]
    """
    assert flow.ndim == 3 and flow.shape[2] == 2, "Flow must have shape [H,W,2]"

    u = flow[:, :, 0]
    v = flow[:, :, 1]
    mag = np.sqrt(u ** 2 + v ** 2)

    # --- Tạo mask theo ngưỡng magnitude ---
    if T_mag is not None:
        mask = (mag >= T_mag)
    else:
        mask = np.ones_like(mag, dtype=bool)

    # --- Lọc vùng nhỏ (nếu có min_area) ---
    if min_area > 0 and max_area > 0:
        mask_uint = (mask.astype(np.uint8) * 255)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint, connectivity=8)
        cleaned = np.zeros_like(mask_uint)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # if min_area <= area <= max_area:  # 🔹 chỉ giữ vùng trong khoảng
            if min_area <= area :
                cleaned[labels == i] = 255
        mask = (cleaned > 0)

    # --- Áp dụng mask ---
    filtered_flow = np.zeros_like(flow)
    filtered_flow[..., 0] = u * mask
    filtered_flow[..., 1] = v * mask

    return filtered_flow
