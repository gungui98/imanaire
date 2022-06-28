import numpy as np
import cv2

import glob
import os

import cv2
import numpy as np
import SimpleITK as itk


def read_mhd(img_path):
    image_itk = itk.ReadImage(img_path)
    image = itk.GetArrayFromImage(image_itk)
    return np.squeeze(image)


def create_mapping(rf_width, rf_height):
    total_angle = 75 * np.pi / 180

    # horizontal center of the image
    half_width = rf_width / 2.0


    x_axis = np.arange(rf_width, dtype=np.float32)
    y_axis = np.arange(rf_height, dtype=np.float32)

    map_x_, map_y_ = np.meshgrid(x_axis, y_axis)

    fy = map_y_
    fx = map_x_ - half_width

    # distance from the point x,y to the center of the transducer, located in -shift_y,half_width
    r = np.sqrt(fy ** 2 + fx ** 2.0)

    # angle of the vector (y, x-half_width) against the half center line
    angle_ = np.arctan2(fx, fy)

    # Invalid values are OK here. cv::remap checks for them and assigns them a special color.
    map_x_ = r * 1.5 / rf_width * rf_height
    map_y_ = ((angle_ - (-total_angle / 2)) / total_angle) * rf_width

    return map_y_, map_x_


def rainbow_image(image_shape):
    """
    Generates a rainbow image of the given shape.
    """
    image = np.zeros(image_shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = (i * 256 / image.shape[0], j * 256 / image.shape[1], 0)
    return image


def gen_noise_image(h, w):
    cone_size = [600, 400]
    xi, zi = create_mapping(cone_size[0], cone_size[1])
    image = np.random.randint(0, 256, (400, 400), dtype=np.uint8)
    # image[:] = [255, 0, 0]
    image = cv2.resize(image, (cone_size[0], cone_size[1]))
    # blur the image
    image = cv2.blur(image, (15, 1))

    warp = cv2.remap(image, xi, zi, cv2.INTER_LINEAR)
    warp = warp[:, 55:-55]
    warp = cv2.resize(warp, (h, w))
    return warp


# if __name__ == '__main__':
#     raw_folder = "E:/echo_gen_data/training"
#     target_folder = "E:/echo_gen_data/camus_label_crop/trainB"
#     shapes = []
#     for mhd_file in glob.glob("{}/*/*_4CH_sequence.mhd".format(raw_folder)):
#         images = read_mhd(mhd_file)
#         video_name = os.path.dirname(mhd_file).split("/")[-1]
#         target_video_folder = "{}/{}".format(target_folder, video_name)
#         if not os.path.exists(target_video_folder):
#             os.makedirs(target_video_folder)
#         for i in range(images.shape[0]):
#             img = images[i]
#
#             noise_img = gen_noise_image(img.shape[1], img.shape[0])
#             add_noise = cv2.addWeighted(img, 0.5, noise_img, 0.5, 0)
#             if [img.shape[0], img.shape[1]] not in shapes:
#                 shapes.append([img.shape[0], img.shape[1]])
#             cv2.imshow('image', img)
#             cv2.imshow('warp', noise_img)
#             cv2.imshow('add_warp', add_noise)
#             cv2.waitKey(0)
#             break
#     print(shapes)
#             # img = cv2.resize(img, (512, 512))
#             # img_name = "{}/file_{:04d}.jpg".format(target_video_folder, i)
#             # cv2.imwrite(img_name, img)



if __name__ == '__main__':
    # create rainbow color map
    # image = rainbow_image((512, 512, 3))
    # create random noise image
    for patient_id in range(1, 100):
        reference_image = cv2.imread(f'E:/echo_gen_data/camus_sequence/patient{patient_id:04d}/images/0000.jpg', 0)
        print(f"E:/echo_gen_data/camus_sequence/patient{patient_id:04d}/images/0000.jpg")
        reference_image = cv2.resize(reference_image, (512, 512))
        xi, zi = create_mapping(600, 400)
        image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        # image[:] = [255, 0, 0]
        image = cv2.resize(image, (600, 400))
        # blur the image
        image = cv2.blur(image, (15, 1))

        warp = cv2.remap(image, xi, zi, cv2.INTER_LINEAR)
        warp = warp[:, 18:-18]
        warp = cv2.resize(warp, (512, 512))
        add_warp = cv2.addWeighted(reference_image, 0.5, warp, 0.5, 0)
        cv2.imshow('image', image)
        cv2.imshow('warp', warp)
        cv2.imshow('add_warp', add_warp)
        cv2.waitKey(0)
