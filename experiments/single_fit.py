"""
fit model to single datapoint
"""
import cv2
import torch
from torch import nn
from imaginaire.config import Config
from imaginaire.generators.unet import UNet
from imaginaire.losses import PerceptualLoss
from imaginaire.model_utils.fs_vid2vid import concat_frames
from imaginaire.utils.dataset import get_train_and_val_dataloader
from imaginaire.utils.trainer import get_model_optimizer_and_scheduler
from imaginaire.utils.visualization import tensor2im


def get_data_t(data, net_G_output, data_prev, t):
    r"""Get data at current time frame given the sequence of data.

    Args:
        data (dict): Training data for current iteration.
        net_G_output (dict): Output of the generator (for previous frame).
        data_prev (dict): Data for previous frame.
        t (int): Current time.
    """
    label = data['label'][:, t]
    image = data['images'][:, t]

    if data_prev is not None:
        # Concat previous labels/fake images to the ones before.
        num_frames_G = 5
        prev_labels = concat_frames(data_prev['prev_labels'],
                                    data_prev['label'], num_frames_G - 1)
        prev_images = concat_frames(
            data_prev['prev_images'],
            net_G_output['fake_images'].detach(), num_frames_G - 1)
    else:
        prev_labels = prev_images = None

    data_t = dict()
    data_t['label'] = label
    data_t['image'] = image
    data_t['prev_labels'] = prev_labels
    data_t['prev_images'] = prev_images
    data_t['real_prev_image'] = data['images'][:, t - 1] if t > 0 else None
    return data_t


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        numel = torch.numel(diff)
        numel = numel * numel
        error = torch.sqrt(diff * diff/numel + self.eps)
        loss = torch.sum(error)
        return loss


class LossFunc:
    def __init__(self):
        self.perceptual = PerceptualLoss().cuda()
        self.l1 = torch.nn.MSELoss()
        self.l1_charbonnier = L1_Charbonnier_loss()

    def get_loss(self, outputs, targets):
        loss = self.l1(outputs, targets)
        loss += self.l1_charbonnier(outputs, targets)
        # loss += self.perceptual(outputs, targets)
        return loss


if __name__ == '__main__':
    cfg = Config('configs/vid2vid_echo_windows.yaml')

    train_data_loader, val_data_loader = get_train_and_val_dataloader(cfg, 0)
    net_G, net_D, opt_G, opt_D, sch_G, sch_D = get_model_optimizer_and_scheduler(cfg, seed=0)
    # net_G = UNet(4, 3).cuda()
    # opt_G = torch.optim.Adam(net_G.parameters(), lr=cfg.gen_opt.lr)
    # data_sample = next(iter(train_data_loader))
    # torch.save(data_sample, 'data_sample.pth')
    data_sample = torch.load('data_sample.pth')
    data_sample['images'] = data_sample['images'].cuda()
    data_sample['label'] = data_sample['label'].cuda()
    sequence_length = data_sample['images'].shape[1]
    loss_func = LossFunc()
    while True:
        output = []
        losses = []
        out_prev = None
        for t in range(sequence_length):
            data_t = {}
            if t == 0:
                out_prev = None
            data_t['label'] = data_sample["label"][:, t]
            data_t['image'] = data_sample["images"][:, t]
            # data_t['prev_labels'] = data_sample["label"][:, t - 1:t] if t > 0 else None
            # data_t['prev_images'] = data_sample["images"][:, t - 1:t] if t > 0 else None
            data_t['prev_labels'] = data_sample["label"][:, t - 1:t] if t > 0 else None
            data_t['prev_images'] = out_prev
            net_G_output = net_G(data_t)
            out_prev = net_G_output['fake_images'][None, ...].detach()
            loss = loss_func.get_loss(net_G_output['fake_images'], data_t['image'])
            # loss += loss_func.get_loss(net_G_output['fake_image'], data_t['image'])
            # backward and optimize
            opt_G.zero_grad()
            loss.backward()
            opt_G.step()
            output.append(net_G_output['fake_images'].detach())
            losses.append(loss.detach())
        print("Loss: {}".format(torch.mean(torch.stack(losses))))
        output = torch.cat(output, dim=0)
        video = tensor2im(data_sample["images"])[0]
        output = tensor2im(output[None, ...])[0]
        for frame_idx in range(len(video)):
            # cv2.imshow("video", video[frame_idx])
            cv2.imshow("pred", output[frame_idx])
            cv2.waitKey(100)
