import glob
import os, shutil
import random

if __name__ == '__main__':
    random.seed(0)
    patients = glob.glob("E:/echo_gen_data/camus_cityscape_format/train/images/*") + \
               glob.glob("E:/echo_gen_data/camus_cityscape_format/test/images/*")
    destination_folder = "E:/echo_gen_data/camus_cityscape_format_shuffle"
    train_portion, test_portion, val_portion = 0.8, 0.1, 0.1
    patients = random.sample(patients, len(patients))
    train_patients = patients[:int(len(patients) * train_portion)]
    test_patients = patients[int(len(patients) * train_portion):int(len(patients) * (train_portion + test_portion))]
    val_patients = patients[int(len(patients) * (train_portion + test_portion)):]
    print("train: {}".format(len(train_patients)))
    print("test: {}".format(len(test_patients)))
    print("val: {}".format(len(val_patients)))
    for patient in train_patients:
        patient_name = patient.split("\\")[-1]
        src_image_path = patient
        src_mask_path = patient.replace("images", "seg_maps")
        dst_img_path = "{}/train/images/{}".format(destination_folder, patient_name)
        dst_msk_path = "{}/train/seg_maps/{}".format(destination_folder, patient_name)
        if not os.path.exists(dst_img_path):
            os.makedirs(dst_img_path)
        if not os.path.exists(dst_msk_path):
            os.makedirs(dst_msk_path)
        for image in glob.glob("{}/*.jpg".format(src_image_path)):
            image_name = image.split("\\")[-1]
            shutil.copy(image, "{}/{}".format(dst_img_path, image_name))
        for image in glob.glob("{}/*.png".format(src_mask_path)):
            image_name = image.split("\\")[-1]
            shutil.copy(image, "{}/{}".format(dst_msk_path, image_name))
            
    for patient in test_patients:
        patient_name = patient.split("\\")[-1]
        src_image_path = patient
        src_mask_path = patient.replace("images", "seg_maps")
        dst_img_path = "{}/test/images/{}".format(destination_folder, patient_name)
        dst_msk_path = "{}/test/seg_maps/{}".format(destination_folder, patient_name)
        if not os.path.exists(dst_img_path):
            os.makedirs(dst_img_path)
        if not os.path.exists(dst_msk_path):
            os.makedirs(dst_msk_path)
        for image in glob.glob("{}/*.jpg".format(src_image_path)):
            image_name = image.split("\\")[-1]
            shutil.copy(image, "{}/{}".format(dst_img_path, image_name))
        for image in glob.glob("{}/*.png".format(src_mask_path)):
            image_name = image.split("\\")[-1]
            shutil.copy(image, "{}/{}".format(dst_msk_path, image_name))

    for patient in val_patients:
        patient_name = patient.split("\\")[-1]
        src_image_path = patient
        src_mask_path = patient.replace("images", "seg_maps")
        dst_img_path = "{}/val/images/{}".format(destination_folder, patient_name)
        dst_msk_path = "{}/val/seg_maps/{}".format(destination_folder, patient_name)
        if not os.path.exists(dst_img_path):
            os.makedirs(dst_img_path)
        if not os.path.exists(dst_msk_path):
            os.makedirs(dst_msk_path)
        for image in glob.glob("{}/*.jpg".format(src_image_path)):
            image_name = image.split("\\")[-1]
            shutil.copy(image, "{}/{}".format(dst_img_path, image_name))
        for image in glob.glob("{}/*.png".format(src_mask_path)):
            image_name = image.split("\\")[-1]
            shutil.copy(image, "{}/{}".format(dst_msk_path, image_name))