import numpy as np
import cv2


def create_mapping(rf_width, rf_height):
    total_angle = 90 * np.pi / 180
    scan_converted = np.zeros((rf_height, rf_width))

    # horizontal center of the image
    half_width = rf_width / 2.0

    map_x = np.zeros(scan_converted.shape, dtype=np.float32)
    map_y = np.zeros(scan_converted.shape, dtype=np.float32)

    for y in range(rf_height):
        for x in range(rf_width):
            fy = y
            fx = x - half_width

            # distance from the point x,y to the center of the transducer, located in -shift_y,half_width
            r = np.sqrt(fy ** 2 + fx ** 2.0)

            # angle of the vector (y, x-half_width) against the half center line
            angle = np.arctan2(fx, fy)

            # Invalid values are OK here. cv::remap checks for them and assigns them a special color.
            map_x[y][x] = r * 1.5 / rf_width * rf_height
            map_y[y][x] = ((angle - (-total_angle / 2)) / total_angle) * rf_width

    return map_y, map_x


def rainbow_image(image_shape):
    """
    Generates a rainbow image of the given shape.
    """
    image = np.zeros(image_shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = (i * 256 / image.shape[0], j * 256 / image.shape[1], 0)
    return image


if __name__ == '__main__':
    # create rainbow color map
    # image = rainbow_image((512, 512, 3))
    # create random noise image
    reference_image = cv2.imread('E:/echo_gen_data/camus_sequence/patient0001/images/0000.jpg',0)
    reference_image = cv2.resize(reference_image, (512, 512))
    xi, zi = create_mapping(600, 400)
    for i in range(1000):
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
        cv2.waitKey(100)
