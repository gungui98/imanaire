import cv2
import numpy as np

from imaginaire.utils.speckle import gen_noise_image

if __name__ == '__main__':
    # read mp4 file
    video_name = "demo.mp4"
    cap = cv2.VideoCapture(video_name)
    noise_images = [gen_noise_image(512, 512) for _ in range(10)]
    video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)
    cap.release()
    while True:
        for i in range(len(video_frames)):
            # warp the image
            # frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            frame = video_frames[i]
            h, w = frame.shape[:2]
            predicted_frame = frame[:, w*2//4:w//4*3]
            noise_frame = noise_images[np.random.randint(0, len(noise_images))]
            noise_frame = cv2.cvtColor(noise_frame, cv2.COLOR_GRAY2BGR)
            add_noise = cv2.addWeighted(predicted_frame, 0.8, noise_frame, 0.2, 0)

            cv2.imshow('image', frame)
            cv2.imshow('noise', noise_frame)
            cv2.imshow('add_warp', add_noise)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break