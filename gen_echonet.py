"""EchoNet-Dynamic Dataset."""

import pathlib
import os
import collections

import cv2
import numpy as np
import torch.utils.data
from tqdm import tqdm


def loadvideo(filename: str) -> np.ndarray:
    """Loads a video from a file.
    Args:
        filename (str): filename of video
    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.
    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame

    return v


class Echo(torch.utils.data.Dataset):
    """EchoNet-Dynamic Dataset.
    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {"train", "val", "test", "external_test"}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):

        self.folder = pathlib.Path(root)
        self.split = split
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if split == "external_test":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            with open(self.folder / "FileList.csv") as f:
                self.header = f.readline().strip().split(",")
                filenameIndex = self.header.index("FileName")
                splitIndex = self.header.index("Split")

                for line in f:
                    lineSplit = line.strip().split(',')

                    fileName = lineSplit[filenameIndex] + ".avi"
                    fileMode = lineSplit[splitIndex].lower()

                    if split in ["all", fileMode] and os.path.exists(self.folder / "Videos" / fileName):
                        self.fnames.append(fileName)
                        self.outcome.append(lineSplit)

            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(self.folder / "VolumeTracings.csv") as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def _reshape_points(self, points_1, points_2, target_size):
        points_1 = points_1 * target_size[0] / 112
        points_2 = points_2 * target_size[1] / 112

        mask_1 = cv2.fillPoly(np.zeros(target_size, dtype=np.uint8), [points_1.astype(np.int32)], 255)
        mask_2 = cv2.fillPoly(np.zeros(target_size, dtype=np.uint8), [points_2.astype(np.int32)], 255)
        # cv2.imshow("mask1", mask_1)
        # cv2.imshow("mask2", mask_2)

        if mask_1.sum() < mask_2.sum():
            mask_1, mask_2 = mask_2, mask_1

        cnt_1 = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2][0].squeeze()
        # for every point in cnt_1, find the closest point in cnt_2 which has 255 value
        mask2_idx = np.where(mask_2 == 255)
        mask2_idx = [mask2_idx[1], mask2_idx[0]]
        cnt_2 = []
        for i in range(cnt_1.shape[0]):
            min_idx = np.argmin((cnt_1[i][0] - mask2_idx[0]) ** 2 + (cnt_1[i][1] - mask2_idx[1]) ** 2)
            cnt_2.append([mask2_idx[0][min_idx], mask2_idx[1][min_idx]])
        cnt_2 = np.array(cnt_2)
        # draw cnt2
        # vis_cnt_2 = cv2.drawContours(np.zeros_like(mask_2), [cnt_2], -1, 255, -1)
        # cv2.imshow("cnt2", vis_cnt_2)
        # cv2.waitKey(0)

        return cnt_1.astype(np.float32), cnt_2.astype(np.float32)

    def _interpolate_mask(self, points_1, points_2, num_frames, target_size=(512, 512)):
        """
        interpolate list of contour point pairs to create a sequence of contour points
        """
        points_1, points_2 = self._reshape_points(points_1, points_2, target_size)
        # cv2.imshow("point 1", cv2.fillPoly(np.zeros(target_size, dtype=np.uint8), [points_1.astype(np.int32)], 255))
        # cv2.imshow("point 2", cv2.fillPoly(np.zeros(target_size, dtype=np.uint8), [points_2.astype(np.int32)], 255))
        # cv2.waitKey(0)
        x1, y1, x2, y2 = points_1[:, 0], points_1[:, 1], points_2[:, 0], points_2[:, 1]

        x_interp = np.linspace(x1, x2, num_frames)
        y_interp = np.linspace(y1, y2, num_frames)
        points = np.stack([x_interp, y_interp], axis=-1)
        mask = []
        for i in range(num_frames):
            mask.append(cv2.drawContours(np.zeros(target_size, dtype=np.uint8), [points[i].astype(np.int32)], -1, 1, -1))
        return mask

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "external_test":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "clinical_test":
            video = os.path.join(self.folder, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.folder, "Videos", self.fnames[index])

        # Load video into np.array
        video = loadvideo(video)

        key = self.fnames[index]
        start_frame_idx = np.int(self.frames[key][0])
        end_frame_idx = np.int(self.frames[key][-1])

        if end_frame_idx < start_frame_idx:
            video = video[end_frame_idx:start_frame_idx][::-1]
        else:
            video = video[start_frame_idx:end_frame_idx]
        # Gather targets
        target = []

        for t in ["SmallTrace", "LargeTrace"]:
            if t == "LargeTrace":
                t = self.trace[key][self.frames[key][-1]]
            else:
                t = self.trace[key][self.frames[key][0]]
            x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
            x = np.concatenate((x1[1:], np.flip(x2[1:])))
            y = np.concatenate((y1[1:], np.flip(y2[1:])))
            target.append(np.stack((x, y), axis=1))

        masks = self._interpolate_mask(target[0], target[1],
                                       len(video))  # no neeed to handle end_frame_idx < start_frame_idx
        # if end_frame_idx < start_frame_idx:
        #     print("error")
        #     for i in range(len(masks)):
        #         vis_mask = cv2.cvtColor(np.uint8(masks[i] * 255.0), cv2.COLOR_GRAY2BGR)
        #         frame = cv2.resize(video[i], (512, 512))
        #         cv2.imshow("img", cv2.addWeighted(frame, 0.5, vis_mask, 0.5, 0))
        #         cv2.waitKey(100)

        return key, video, masks

    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


if __name__ == '__main__':

    for data_type in ["train", "test", "val"]:
        data = Echo("E:/echo_gen_data/EchoNet-Dynamic", split=data_type,
                    target_type=["LargeTrace", "SmallTrace"])
        for i in tqdm(range(len(data))):
            key, video, masks = data.__getitem__(i)
            key = key.split(".")[0]
            tgt_image_path = "E:/echo_gen_data/echonet_cityscape/{}/images/{}".format(data_type, key)
            tgt_mask_path = "E:/echo_gen_data/echonet_cityscape/{}/seg_maps/{}".format(data_type, key)
            if not os.path.exists(tgt_image_path):
                os.makedirs(tgt_image_path, exist_ok=True)
            if not os.path.exists(tgt_mask_path):
                os.makedirs(tgt_mask_path, exist_ok=True)
            for frame_idx in range(len(video)):
                frame = video[frame_idx]
                frame = cv2.resize(frame, (256, 256))
                mask = masks[frame_idx]
                mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(tgt_image_path, "{:04d}.jpg".format(frame_idx)), frame)
                cv2.imwrite(os.path.join(tgt_mask_path, "{:04d}.png".format(frame_idx)), mask)
