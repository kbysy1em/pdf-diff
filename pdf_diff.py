import numpy as np
import cv2
import pdf2image
import time
from matplotlib import pyplot as plt

try:
    # images will be a list of PIL Image representing each page of the PDF document.
    images1 = pdf2image.convert_from_path('00.pdf', grayscale=True, dpi=600)
    img1 = np.array(images1[0], dtype=np.uint8)
   
    plt.imshow(img1[200:600, 200:600])
    plt.show()
    # x1, y1 = map(int, input('origin1').split())
    x1, y1 = 244, 116

    images2 = pdf2image.convert_from_path('01.pdf', grayscale=True, dpi=600)
    img2 = np.array(images2[0], dtype=np.uint8)
    plt.imshow(img2[200:600, 200:600])
    plt.show()
    # x2, y2 = map(int, input('origin2').split())
    x2, y2 = 242, 26

    delta_x = x2 - x1
    delta_y = y2 - y1

    M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    img1 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]), borderValue=255)

    # plt.subplot(121).imshow(img1, vmin=0, vmax=255)
    # plt.subplot(122).imshow(img2, vmin=0, vmax=255)
    # plt.show()

    color_img = cv2.merge((img2, img2, img2))

    # 縦がX(下向きが正)、横がY(右向きが正)
    # イントロゲーションエリアの原点は左上 
    int_height = 100
    int_width = 100
    step_x = 0.5
    step_y = 0.5
    scan_area_ratio_x = 0.25
    scan_area_ratio_y = 0.25
    ite_x = scan_area_ratio_x / step_x  # 初期値
    ite_y = scan_area_ratio_y / step_y  # 初期値

    def current_x():
        return int(ite_x * int_height * step_x)
    def current_y():
        return int(ite_y * int_width * step_y)

    while current_x() + int_height * (scan_area_ratio_x + 1) < img2.shape[0]:
        while current_y() + int_width * (scan_area_ratio_y + 1) < img2.shape[1]:
            template = img2[current_x():current_x() + int_height, current_y():current_y() + int_width]

            method = eval('cv2.TM_CCORR_NORMED')
            # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            #             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

            scan_img = img1[current_x() - int(int_height * scan_area_ratio_x)
                :current_x() + int(int_height * (scan_area_ratio_x + 1)),
                current_y() - int(int_width * scan_area_ratio_y)
                :current_y() + int(int_width * (scan_area_ratio_y + 1))]

            res = cv2.matchTemplate(scan_img, template, method)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val < 0.98:
                part_img = color_img[current_x():current_x() + int_height, current_y():current_y() + int_width]
                white_pixels = (part_img == (255, 255, 255)).all(axis=2)
                part_img[white_pixels] = (255, 99, 71)
                color_img[current_x():current_x() + int_height, current_y():current_y() + int_width] = part_img

            ite_y += 1
        ite_y = scan_area_ratio_y / step_y
        ite_x += 1
        print(current_x())

    plt.subplot(121).imshow(img1, vmin=0, vmax=255, cmap='gray')
    plt.subplot(122).imshow(color_img)
    plt.show()

except:
    import sys
    print('Error: ', sys.exc_info()[0])
    print(sys.exc_info()[1])
    import traceback
    print(traceback.format_tb(sys.exc_info()[2]))

