import glob
import pycocotools.coco as coco
import os
import cv2

dir_name = "../data/"
anot_path = '../data/coco/annotations/person_keypoints_train2017.json'
my_coco = coco.COCO(anot_path)
my_imgs = my_coco.getImgIds()
max_objs = 32
max_kps = 17

for index in range(64115):
    img_id = my_imgs[index]
    file_path_ext = my_coco.loadImgs(img_id)[0]['file_name']
    img_pth_const = '../data/coco/train2017'
    image_path = os.path.join(img_pth_const, file_path_ext)
    img = cv2.imread(image_path)
    if img is None:
        print(f"image path: {image_path}")

print("Done")
