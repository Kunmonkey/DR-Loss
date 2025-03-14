# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data

# from config import config

y_k_size = 6
x_k_size = 6

class BaseDataset(data.Dataset):
    def __init__(self,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        self.mean = mean
        self.std = std
        self.scale_factor = scale_factor
        self.downsample_rate = 1./downsample_rate

        self.files = []

        # 在列维度上，dim 0 -> crop前不同类别在一次训练中的累计数目
        # 在列维度上，dim 1 -> crop后不同类别在一次训练中的累计数目
        # 在列维度上，dim 2 -> crop后不同类别累计像素数/crop前不同类别累计像素数，
        self.obj_cnt = np.array([[0,0,0,0] for i in range(8)],dtype=float)
        self.clock = 0
    def __len__(self):
        return len(self.files)

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        return image

    def label_transform(self, label):
        print(np.unique(np.array(label).astype('int32')))
        return np.array(label).astype('int32')

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label, edge=None):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        # label = self.pad_image(label, h, w, self.crop_size,
        #                        (0,))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))
        if self.pid:
            edge = self.pad_image(edge, h, w, self.crop_size,
                                   (0.0,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        if self.pid:
            edge = edge[y:y+self.crop_size[0], x:x+self.crop_size[1]]
            return image, label, edge
        return image, label

    def multi_scale_aug(self, image, label=None, edge=None,
                        rand_scales=1, rand_crop=True):
        # print(f'ori image size:{label.shape}')

        # self.clock += 1
        # print(f'clock: {self.clock}')

        images = [image]
        # rand_scales = [0.7,1.5]
        for rand_scale in rand_scales:
            long_size = int(self.base_size * rand_scale + 0.5)
            h, w = image.shape[:2]
            if h > w:
                new_h = long_size
                new_w = int(w * long_size / h + 0.5)
            else:
                new_w = long_size
                new_h = int(h * long_size / w + 0.5)
            # print(f'h:{new_h},w:{new_w},long size:{long_size}')
            # image = cv2.resize(image, (new_w, new_h),
            #                    interpolation=cv2.INTER_LINEAR)
            new_image = cv2.resize(image, (new_w, new_h),
                               interpolation=cv2.INTER_LINEAR)
            if not self.pmsa:
                images[0] = new_image
            else:
                images.append(new_image)
        ############ when test or evaluation, annotate the below code
        if self.mode == 'train':
            if label is not None:
                label = cv2.resize(label, (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)
                if edge is not None:
                    edge = cv2.resize(edge, (new_w, new_h),
                                       interpolation=cv2.INTER_NEAREST)
            else:
                return images

            # msa之后，crop前，标签中不同类别，和累计像素值
            # tot_lab, tot_cnt = np.unique(label, return_counts=True)
            # self.obj_cnt[tot_lab,0] += 1
            # self.obj_cnt[tot_lab,3] += tot_cnt / sum(tot_cnt)
            # # print(f'tot lab: {tot_lab}, after:{self.obj_cnt}')
            
            # when training, using rand crop
            if rand_crop:
                if self.pid:
                    images, label, edge = self.rand_crop(images[-1], label, edge=edge)
                    images = [images]
                else:
                    images, label = self.rand_crop(images[-1], label)
                    images = [images]
            # print(f'image size:{image.shape}')

            #     # crop后，标签中不同类别，和累计像素值
            #     crop_lab, crop_cnt = np.unique(label, return_counts=True)
            #     if 255 in crop_lab:
            #         crop_lab = crop_lab[:-1]
            #         crop_cnt = crop_cnt[:-1]
            #     self.obj_cnt[crop_lab,1] += 1
            #     self.obj_cnt[crop_lab,2] += crop_cnt / tot_cnt[np.isin(tot_lab,crop_lab)]
            # # print(f'now {self.obj_cnt}')
        if self.pid:
            return images, label, edge
        return images, label

    def resize_exponent_length(self, image, label=None, exp=5):
        h, w = image.shape[:2]
        new_h = int(h / 2**exp + 0.5) * 2**exp
        new_w = int(w / 2**exp + 0.5) * 2**exp
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            return image, label
        return image
    

    def resize_crop_size(self, image, label=None):
        image = cv2.resize(image, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_CUBIC)
        if label is not None:
            label = cv2.resize(label, (self.crop_size[0], self.crop_size[1]), interpolation=cv2.INTER_NEAREST)
            return image, label
        return image

    def resize_short_length(self, image, label=None, short_length=None, fit_stride=None, return_padding=False):
        h, w = image.shape[:2]
        if h < w:
            new_h = short_length
            new_w = int(w * short_length / h + 0.5)
        else:
            new_w = short_length
            new_h = int(h * short_length / w + 0.5)        
        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = 0, 0
        if fit_stride is not None:
            pad_w = 0 if (new_w % fit_stride == 0) else fit_stride - (new_w % fit_stride)
            pad_h = 0 if (new_h % fit_stride == 0) else fit_stride - (new_h % fit_stride)
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, 
                cv2.BORDER_CONSTANT, value=tuple(x * 255 for x in self.mean[::-1])
            )

        if label is not None:
            label = cv2.resize(
                label, (new_w, new_h),
                interpolation=cv2.INTER_NEAREST)
            if pad_h > 0 or pad_w > 0:
                label = cv2.copyMakeBorder(
                    label, 0, pad_h, 0, pad_w, 
                    cv2.BORDER_CONSTANT, value=self.ignore_label
                )
            if return_padding:
                return image, label, (pad_h, pad_w)
            else:
                return image, label
        else:
            if return_padding:
                return image, (pad_h, pad_w)
            else:
                return image  

    # def random_brightness(self, img):
    #     if not config.TRAIN.RANDOM_BRIGHTNESS:
    #         return img
    #     if random.random() < 0.5:
    #         return img
    #     self.shift_value = config.TRAIN.RANDOM_BRIGHTNESS_SHIFT_VALUE
    #     img = img.astype(np.float32)
    #     shift = random.randint(-self.shift_value, self.shift_value)
    #     img[:, :, :] += shift
    #     img = np.around(img)
    #     img = np.clip(img, 0, 255).astype(np.uint8)
    #     return img

    # def gen_sample(self, image, label,
    #                multi_scale=True, is_flip=True, edge_pad=True, edge_size=4.0):


    #     if multi_scale:
    #         # rand_scale = [0.5 + random.randint(0, self.scale_factor) / 10.0]
    #         # print(rand_scale)
    #         # rand_scale = [1. + random.randint(0, 6) / 10.0]
    #         rand_scale = [0.5,0.75,1,1.25,1.5,1.75]
    #         # rand_scale = [2]
    #         if self.pid:
    #             image, label, edge = self.multi_scale_aug(image, label, edge=edge,
    #                                             rand_scales=rand_scale)
    #         else:
    #             image, label = self.multi_scale_aug(image, label,
    #                                                 rand_scales=rand_scale)
    #         # print(f'rand scale:{rand_scale}')
    #     # print(f'num of image:{len(image)}')
    #     images = []
    #     if not isinstance(image,list):
    #         image = [image]
    #     for i in range(len(image)):
    #         img = self.random_brightness(image[i])
    #         img = self.input_transform(img)
    #         label = self.label_transform(label)
    #         img = img.transpose((2, 0, 1))

    #         if is_flip:
    #             flip = np.random.choice(2) * 2 - 1
    #             img = img[:, :, ::flip]
    #             label = label[:, ::flip]
    #             if self.pid:
    #                 edge = edge[:, ::flip]
    #         images.append(img)
    #     if self.downsample_rate != 1:
    #         label = cv2.resize(
    #             label,
    #             None,
    #             fx=self.downsample_rate,
    #             fy=self.downsample_rate,
    #             interpolation=cv2.INTER_NEAREST
    #         )
    #     if self.pid:
    #         # print(images[0].shape,images[1].shape)
    #         return images, label, edge, self.obj_cnt
    #     return images, label, self.obj_cnt

    def reduce_zero_label(self, labelmap):
        labelmap = np.array(labelmap)
        # encoded_labelmap = labelmap 
        encoded_labelmap = labelmap - 1

        return encoded_labelmap

    def inference(self, config, model, image, flip=False):
        size = image.size()
        print("infer size : ", size)
        pred = model(image)
        # print(f"pred size : {pred[1].size()},len pred:{len(pred)}", )
        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]

        # pred = F.interpolate(
        #     input=pred, size=size[-2:],
        #     mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        # )
        # print(f"new pred size : {pred.size()}")
        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))

            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]

            flip_output = F.interpolate(
                input=flip_output, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = int(self.crop_size[0] * 2.0 / 3.0)
        stride_w = int(self.crop_size[1] * 2.0 / 3.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()
        padvalue = -1.0 * np.array(self.mean) / np.array(self.std)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if max(height, width) <= np.min(self.crop_size):
                new_img = self.pad_image(new_img, height, width,
                                         self.crop_size, padvalue)
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                if height < self.crop_size[0] or width < self.crop_size[1]:
                    new_img = self.pad_image(new_img, height, width,
                                             self.crop_size, padvalue)
                new_h, new_w = new_img.shape[:-1]
                rows = int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        if h1 == new_h or w1 == new_w:
                            crop_img = self.pad_image(crop_img,
                                                      h1-h0,
                                                      w1-w0,
                                                      self.crop_size,
                                                      padvalue)
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1-h0, 0:w1-w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred
