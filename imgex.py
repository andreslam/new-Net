import cv2
import numpy as np
from options import args
import cv2
import random

# def random_crop(img, gt):
#     h, w = img.shape[:2]
#     crop_size = min(h, w) // 2
#     x = random.randint(0, w - crop_size)
#     y = random.randint(0, h - crop_size)
#     crop_img = img[y:y+crop_size, x:x+crop_size]
#     crop_gt = gt[y:y+crop_size, x:x+crop_size]
#     return crop_img, crop_gt
#
#
# 定义随机水平翻转函数
def random_flip(img, gt):
    flip = np.random.randint(0, 2)
    if flip == 1:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)
    return img, gt
#
# for i in range(30):
#     if i==0 :
#         continue
#     img = cv2.imread("G:/OCTA/ROSE/data/ROSE-1/SVC_DVC/train/img/" + str(i) + ".png")
#     gt = cv2.imread("G:/OCTA/ROSE/data/ROSE-1/SVC_DVC/train/gt/" + str(i) + ".tif")
#     # g_img =np.array(img)
#     # g_gt=np.array((gt))
#     img1,gt1=random_flip(img,gt)
    # cv2.imwrite("E:/new/img/" + str(i + 30) + ".png", img1)
    # cv2.imwrite("E:/new/img/" + str(i + 30) + ".png", img1.astype(np.uint8))
    #
    # flipped_img, flipped_gt = random_flip(g_img, g_gt)
    # crops_img = []
    # crops_gt = []
    #
    # crop_img, crop_gt = random_crop(flipped_img, flipped_gt)
    #
    #
    #
    # # 保存增广后的图像
    # cv2.imwrite("E:/new/img/" + str(i+30) + ".png", crop_img)
    # cv2.imwrite("E:/new/gt/" + str(i +30) + ".tif", crop_gt)
# 显示结果
# cv2.imshow('Original', img)
# cv2.imshow('Augmented', flipped_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
def random_crop(img, gt, crop_scale):
    h, w, _ = img.shape
    crop_h = int(h * crop_scale)
    crop_w = int(w * crop_scale)
    x = np.random.randint(0, w - crop_w)
    y = np.random.randint(0, h - crop_h)
    img_crop = img[y:y + crop_h, x:x + crop_w, :]
    gt_crop = gt[y:y + crop_h, x:x + crop_w, :]
    img_crop = cv2.resize(img_crop, (w, h))
    gt_crop = cv2.resize(gt_crop, (w, h))
    return img_crop, gt_crop
for i in range(31):
    if i==0 :
        continue
    if i<=9 :
      img = cv2.imread("G:/OCTA/ROSE/data/ROSE-1/SVC_DVC/train/img/0"+str(i) + ".png")
      gt = cv2.imread("G:/OCTA/ROSE/data/ROSE-1/SVC_DVC/train/gt/0" +str(i) + ".tif")
    else:
        img = cv2.imread("G:/OCTA/ROSE/data/ROSE-1/SVC_DVC/train/img/" + str(i) + ".png")
        gt = cv2.imread("G:/OCTA/ROSE/data/ROSE-1/SVC_DVC/train/gt/" + str(i) + ".tif")
    print(img)
    img1, gt1 = random_flip(img, gt)
    g_img1 = np.array(img1)

    g_gt1 = np.array(gt1)
    print(g_gt1.shape)
    img2, gt2 = random_crop(g_img1, g_gt1, crop_scale=0.8)
    cv2.imwrite("E:/new/img/" + str(i + 30) + ".png", img2)
    cv2.imwrite("E:/new/gt/" + str(i + 30) + ".tif", gt2)