import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import ColorJitter
import torch

class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True,
                 no_eraser_aug=False, resize_or_crop_mode='crop'):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        # self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

        self.asymmetric_color_aug_prob = 0.2

        if no_eraser_aug:
            # we disable eraser aug since no obvious improvement is observed in our experiments
            self.eraser_aug_prob = -1
        else:
            self.eraser_aug_prob = 0.5

    def color_jitter_numpy_uint8(self, img):
        # img: numpy uint8, RGB, shape [H, W, C]
        img_tensor = torch.from_numpy(img.copy())
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()
        img_tensor = img_tensor.to(torch.float32) / 255.0

        jitter = self.photo_aug(img_tensor)

        img_out = img_tensor.clamp(0, 1)
        img_out = (img_out * 255).round().byte()
        # img_out = img_out.to(torch.uint8)
        img_out = img_out.permute(1, 2, 0).cpu().numpy()

        return img_out

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)

            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)

            # image_stack = self.color_jitter_numpy_uint8(image_stack)

            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow, occlusion=None):
        # randomly sample scale
        ht, wd = img1.shape[:2]

        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

            if occlusion is not None:
                occlusion = cv2.resize(occlusion, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

                if occlusion is not None:
                    occlusion = occlusion[:, ::-1]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

                if occlusion is not None:
                    occlusion = occlusion[::-1, :]

        # In case no cropping
        if img1.shape[0] - self.crop_size[0] > 0:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        else:
            y0 = 0
        if img1.shape[1] - self.crop_size[1] > 0:
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        else:
            x0 = 0

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        if occlusion is not None:
            occlusion = occlusion[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            return img1, img2, flow, occlusion

        return img1, img2, flow

    def __call__(self, img1, img2, flow, occlusion=None):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)

        if occlusion is not None:
            img1, img2, flow, occlusion = self.spatial_transform(
                img1, img2, flow, occlusion)
        else:
            img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        if occlusion is not None:
            occlusion = np.ascontiguousarray(occlusion)
            return img1, img2, flow, occlusion

        return img1, img2, flow


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False,
                 no_eraser_aug=False,
                 ):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2

        if no_eraser_aug:
            # we disable eraser aug since no obvious improvement is observed in our experiments
            self.eraser_aug_prob = -1
        else:
            self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)

        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid

class ImageOnlyAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True,
                 no_eraser_aug=False, resize_or_crop_mode = 'crop'
                 ):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)

        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = -1 if no_eraser_aug else 0.5
        self.resize_or_crop_mode = resize_or_crop_mode  # 'crop' | 'resize' | 'random'

    def color_transform(self, img1, img2):
        """ Photometric augmentation """
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return img1, img2

    def spatial_transform(self, img1, img2):
        if self.resize_or_crop_mode == 'random':
            use_resize = np.random.rand() < 0.5
        else:
            use_resize = self.resize_or_crop_mode == 'resize'

        if use_resize:
            # Resize to crop_size
            img1 = cv2.resize(img1, self.crop_size[::-1], interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, self.crop_size[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            # Padding + crop logic
            h, w = img1.shape[:2]
            target_h, target_w = self.crop_size

            # calculate padding
            pad_top = max(0, (target_h - h) // 2)
            pad_bottom = max(0, target_h - h - pad_top)
            pad_left = max(0, (target_w - w) // 2)
            pad_right = max(0, target_w - w - pad_left)

            # Padding 
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                img1 = np.pad(img1, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                             mode='constant', constant_values=0)
                img2 = np.pad(img2, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                             mode='constant', constant_values=0)

            # after padding, random crop
            padded_h, padded_w = img1.shape[:2]
            if padded_h - target_h > 0:
                y0 = np.random.randint(0, padded_h - target_h)
            else:
                y0 = 0
            if padded_w - target_w > 0:
                x0 = np.random.randint(0, padded_w - target_w)
            else:
                x0 = 0

            img1 = img1[y0:y0 + target_h, x0:x0 + target_w]
            img2 = img2[y0:y0 + target_h, x0:x0 + target_w]

            # ensure contiguous
            img1 = np.ascontiguousarray(img1)
            img2 = np.ascontiguousarray(img2)

        return img1, img2

    def safe_pad_or_crop(self, img, target_size):
        h, w = img.shape[:2]
        th, tw = target_size
        out = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
        y_start = max(0, (th - h) // 2)
        x_start = max(0, (tw - w) // 2)
        out[y_start:y_start+h, x_start:x_start+w] = img
        # return out
        return np.ascontiguousarray(out)

    def __call__(self, img1, img2):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2 = self.spatial_transform(img1, img2)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)

        return img1, img2

def is_contiguous(arr):
    """Return True if `arr` is contiguous in either C or Fortran order."""
    return arr.flags['C_CONTIGUOUS'] or arr.flags['F_CONTIGUOUS']