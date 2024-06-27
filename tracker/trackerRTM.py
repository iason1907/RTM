import cv2
import torch
import numpy as np
import torch.nn as nn
from utils.util import util
import torch.nn.functional as F
from configs.config import TrackerConfig
import torchvision.transforms as transforms
from utils.custom_transforms import ToTensor
from configs.config import config
from toolkit.got10k.trackers import Tracker
from tracker.network import SiamRPNRTMNet
from utils.data_loader import TrackerDataLoader
from PIL import Image, ImageOps, ImageStat, ImageDraw
from utils.torchvotrt import noise_handler_tensor

class TrackerSiamRTM(Tracker):
    def __init__(self, params, model_path = None, distortion='original', **kargs):
        super(TrackerSiamRTM, self).__init__(name='SiamRTM', is_deterministic=True)
        self.distortion = distortion
        self.model = SiamRPNRTMNet()

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        checkpoint = torch.load(model_path, map_location = self.device)
        #print("1")
        if 'model' in checkpoint.keys():
            self.model.load_state_dict(torch.load(model_path, map_location = self.device)['model'])
        else:
            self.model.load_state_dict(torch.load(model_path, map_location = self.device))
        if self.cuda:
            self.model = self.model.cuda()
        self.model.eval()
        self.transforms = transforms.Compose([
            ToTensor()
        ])

        valid_scope = 2 * config.valid_scope + 1
        self.anchors = util.generate_anchors(   config.total_stride,
                                                config.anchor_base_size,
                                                config.anchor_scales,
                                                config.anchor_ratios,
                                                valid_scope)
        self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                              [config.anchor_num, 1, 1]).flatten()

        self.data_loader = TrackerDataLoader()
    
    def evalSIDD(self, image):
        image_tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float().cuda()
        return self.model.eval_auto(image_tensor).squeeze(0).permute(1,2,0).data.cpu().numpy().astype(np.uint8)

    def init(self, frame, bbox):
        frame = np.asarray(frame)

        self.pos = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2])  
        self.target_sz = np.array([bbox[2], bbox[3]])
        self.bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]])

        self.origin_target_sz = np.array([bbox[2], bbox[3]])
        self.img_mean = np.mean(frame, axis=(0, 1))

        exemplar_img, _, _ = self.data_loader.get_exemplar_image(   frame,
                                                                    self.bbox,
                                                                    config.template_img_size,
                                                                    config.context_amount,
                                                                    self.img_mean)

        exemplar_img = (self.transforms(exemplar_img)[None, :, :, :])
        if self.cuda:
            self.model.track_init(exemplar_img.cuda())
        else:
            self.model.track_init(exemplar_img)

    def update(self, frame):
        frame = np.asarray(frame)

        instance_img, _, _, scale_x = self.data_loader.get_instance_image(  frame,
                                                                            self.bbox,
                                                                            config.template_img_size,
                                                                            config.detection_img_size,
                                                                            config.context_amount,
                                                                            self.img_mean)

        if self.cuda:
            instance_img = (noise_handler_tensor(self.transforms(instance_img)[None, :, :, :], self.distortion)).cuda()
        else:
            instance_img = (noise_handler_tensor(self.transforms(instance_img)[None, :, :, :], self.distortion))

        pred_score, pred_regression = self.model.track(instance_img)
        pred_conf   = pred_score.reshape(-1, 2, config.size ).permute(0, 2, 1)
        pred_offset = pred_regression.reshape(-1, 4, config.size ).permute(0, 2, 1)

        delta = pred_offset[0].cpu().detach().numpy()
        box_pred = util.box_transform_inv(self.anchors, delta)
        score_pred = F.softmax(pred_conf, dim=2)[0, :, 1].cpu().detach().numpy()

        s_c = util.change(util.sz(box_pred[:, 2], box_pred[:, 3]) / (util.sz_wh(self.target_sz * scale_x)))  # scale penalty
        r_c = util.change((self.target_sz[0] / self.target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)
        pscore = penalty * score_pred
        pscore = pscore * (1 - config.window_influence) + self.window * config.window_influence
        best_pscore_id = np.argmax(pscore)
        target = box_pred[best_pscore_id, :] / scale_x

        lr = penalty[best_pscore_id] * score_pred[best_pscore_id] * config.lr_box

        res_x = np.clip(target[0] + self.pos[0], 0, frame.shape[1])
        res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[0])

        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])

        bbox = np.array([res_x, res_y, res_w, res_h])
        self.bbox = (
            np.clip(bbox[0], 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64))

        res_x = res_x - res_w/2 
        res_y = res_y - res_h/2
        bbox = np.array([res_x, res_y, res_w, res_h])
        return bbox
