import glob
import os.path

from imaginaire.utils.speckle import gen_noise_image, cone_mask

import cv2
import numpy as np
import matplotlib.pyplot as plt


def simulate(fake, real):
    # fake = pad_img(fake)
    real = cv2.resize(real, (fake.shape[1], fake.shape[0]))

    cv2.imshow('fake', fake)
    cv2.imshow('real', real)

    fake_fft = np.fft.fftn(fake)
    real_fft = np.fft.fftn(real)

    fake_mag = np.abs(fake_fft)
    fake_pha = np.angle(fake_fft)

    real_mag = np.abs(real_fft)
    real_pha = np.angle(real_fft)

    # plot(np.log(fake_mag))
    # plot(np.log(real_mag))
    # plot(fake_pha)
    # plot(real_pha)
    sigma_y = 1.0
    sigma_x = 1.0
    sigma = [sigma_y, sigma_x]

    # real_pha = sp.ndimage.gaussian_filter(real_pha, sigma, mode='constant')

    def blend(a_mag, a_pha, b_pha, alpha):
        c_mag = a_mag
        c_pha = alpha * a_pha + (1 - alpha) * b_pha
        c = c_mag * np.exp(1j * c_pha)
        z = np.fft.ifftn(c).real
        return np.uint8(z)

    for alpha in np.linspace(0, 1, 4):
        z = blend(fake_mag, fake_pha, real_pha, alpha)
        cv2.imshow(f'simulate{alpha}', z)


def show_phase(name, img):
    mfft = np.fft.fftn(img)
    mag = np.abs(mfft)
    pha = np.angle(mfft)

    x = np.log(mag)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    cv2.imshow(f'{name} mag', x)
    # plot(np.log(mag))
    cv2.imshow(f'{name} pha', pha)


def label2weightmap(label):
    weight_map = np.zeros_like(label, dtype=np.float32)
    weight_map[label == 0] = 0  # background
    weight_map[label == 1] = 0.2  # lv blood pool
    weight_map[label == 2] = 1  # lv myo
    weight_map[label == 3] = 0.2  # la blood pool
    # for i in range(4):
    #     cv2.imshow("wind", (label == i).astype(np.uint8) * 255)
    #     cv2.waitKey(0)

    # label = np.uint8(label * 255)
    # label = cv2.dilate(label, np.ones((13, 13)))

    # label = label.astype(np.float32)
    weight_map = cv2.GaussianBlur(weight_map, (151, 151), 50)
    # cv2.imshow('label', np.uint8(weight_map * 255.0))

    # label = label * 255
    # dialated = cv2.dilate(label, np.ones((3, 3)))
    # scale to [0.2, 0.8]
    weight_map = (weight_map - np.min(weight_map)) / (np.max(weight_map) - np.min(weight_map))
    weight_map = weight_map * 0.6 + 0.4
    return weight_map


def get_background_hist(predicted_frame, cone_mask, label_frame):
    label_frame = cv2.resize(label_frame, (predicted_frame.shape[1], predicted_frame.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
    label_frame = (label_frame > 0).astype(np.uint8) * 255
    label_frame = cv2.dilate(label_frame, np.ones((41, 41)))
    background = np.logical_and(cone_mask, (255 - label_frame))
    background_intensity = predicted_frame[background]
    hist = np.histogram(background_intensity, bins=np.arange(0, 257, 1))
    # normalize
    hist = hist[0] / np.sum(hist[0])
    # plot histogram
    return hist


if __name__ == '__main__':
    # read mp4 file
    video_name = "output/u000_model000_r001_v001_i000_small_sample.mp4"
    label_folder = "C:/Users/admin/PycharmProjects/" \
                   "pseudo-image-extraction/data/synthetic_data/labels/" + os.path.basename(video_name).replace(".mp4",
                                                                                                                "")
    label_paths = glob.glob(os.path.join(label_folder, "*.png"))
    cap = cv2.VideoCapture(video_name)
    video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    cap.release()
    # noise_images = [gen_noise_image(512, 512) for _ in range(len(video_frames))]
    cone_mask = cone_mask(512, 512)
    add_noises = []
    label_frames = []
    while True:
        for i in range(len(video_frames)):
            # warp the image
            # frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            frame = video_frames[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]
            predicted_frame = frame[:, w * 2 // 3:]
            label_frame = cv2.imread(label_paths[i], 0)
            weigt_map = label2weightmap(label_frame)
            weigt_map = cv2.resize(weigt_map, (predicted_frame.shape[1], predicted_frame.shape[0]))
            background_hist = get_background_hist(predicted_frame, cone_mask, label_frame)

            # noise_frame = noise_images[np.random.randint(0, len(noise_images))]
            noise_frame = gen_noise_image(predicted_frame.shape[0], predicted_frame.shape[1], background_hist)
            # noise_frame = cv2.cvtColor(noise_frame, cv2.COLOR_GRAY2BGR)
            # gausian filter the image
            # predicted_frame = cv2.GaussianBlur(predicted_frame, (5, 5), 0)

            # im_clone = cv2.seamlessClone(predicted_frame, gen_noise_image(512, 512),
            #                              cv2.resize((label_frame > 0).astype(np.uint8), (512, 512),
            #                                         interpolation=cv2.INTER_NEAREST),
            #                              (predicted_frame.shape[0] // 2, predicted_frame.shape[1] // 2), cv2.MIXED_CLONE)
            # cv2.imshow("im_clone", im_clone)
            # cv2.waitKey(0)

            predicted_frame_ = predicted_frame.astype(np.float32) / 255.0
            noise_frame_ = noise_frame.astype(np.float32) / 255.0
            predicted_frame_ = predicted_frame_ ** 1.2
            add_noise = predicted_frame_ * weigt_map + noise_frame_ * (1 - weigt_map)
            add_noise = add_noise * 256.0
            add_noise = add_noise.astype(np.uint8)
            # increase contrast
            # add_noise = cv2.addWeighted(add_noise, 1.1, add_noise, 0, 0)

            # add_noise = cv2.addWeighted(predicted_frame, 0.8, noise_frame, 0.2, 0)

            # simulate(noise_frame, predicted_frame)
            # cv2.waitKey(0)
            add_noises.append(add_noise)
            label_frames.append(frame[:, :w * 1 // 3])
        while True:
            for i in range(len(add_noises)):
                cv2.imshow("", add_noises[i])
                cv2.imshow("label", label_frames[i])

                cv2.waitKey(100)

        # cv2.imshow('image', frame)
        # cv2.imshow('noise', noise_frame)
        # cv2.imshow('add_noise', add_noise)
        # cv2.imshow('label_frame', weigt_map)
        # if cv2.waitKey(50) & 0xFF == ord('q'):
        #     break
