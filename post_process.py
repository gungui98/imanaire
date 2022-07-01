from imaginaire.utils.speckle import gen_noise_image

import cv2
import numpy as np


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
    #real_pha = sp.ndimage.gaussian_filter(real_pha, sigma, mode='constant')

    def blend(a_mag, a_pha, b_pha, alpha):
        c_mag = a_mag
        c_pha = alpha*a_pha + (1-alpha)*b_pha
        c = c_mag*np.exp(1j*c_pha)
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
    x = (x-np.min(x)) / (np.max(x)-np.min(x))
    cv2.imshow(f'{name} mag', x)
    #plot(np.log(mag))
    cv2.imshow(f'{name} pha', pha)


if __name__ == '__main__':
    # read mp4 file
    video_name = "demo.mp4"
    cap = cv2.VideoCapture(video_name)
    video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    cap.release()
    noise_images = [gen_noise_image(512, 512) for _ in range(len(video_frames))]
    for i in range(len(video_frames)):
        cv2.imwrite("./noise/{}.png".format(i), noise_images[i])
    while True:
        for i in range(len(video_frames)):
            # warp the image
            # frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            frame = video_frames[i]
            h, w = frame.shape[:2]
            predicted_frame = frame[:, w*2//4:w//4*3]
            label_frame = frame[:, w//4:w//4*3]
            noise_frame = noise_images[np.random.randint(0, len(noise_images))]
            noise_frame = cv2.cvtColor(noise_frame, cv2.COLOR_GRAY2BGR)
            # gausian filter the image
            predicted_frame = cv2.GaussianBlur(predicted_frame, (5, 5), 0)
            add_noise = cv2.addWeighted(predicted_frame, 0.8, noise_frame, 0.2, 0)

            # simulate(noise_frame, predicted_frame)
            # cv2.waitKey(0)

            cv2.imshow('image', frame)
            cv2.imshow('noise', noise_frame)
            cv2.imshow('add_warp', add_noise)
            cv2.imshow('label_frame', label_frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break