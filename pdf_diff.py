import numpy as np
import cv2
import pdf2image
import time
from matplotlib import pyplot as plt

try:
    # images will be a list of PIL Image representing each page of the PDF document.
    images1 = pdf2image.convert_from_path('00.pdf', grayscale=True, dpi=600)
    img1 = np.array(images1[0], dtype=np.uint8)
    images2 = pdf2image.convert_from_path('01.pdf', grayscale=True, dpi=600)
    img2 = np.array(images2[0], dtype=np.uint8)

    color_img = cv2.merge((img2, img2, img2))

    # 縦がX(下向きが正)、横がY(右向きが正)
    # イントロゲーションエリアの原点は左上 
    int_height = 100
    int_width = 100
    ite_x = 2
    ite_y = 2
    overlap = 0
    int_x = ite_x * int_height
    int_y = ite_y * int_width

    while int_x + int_height * 3 <= img2.shape[0]:
    # if True:
    #     ite_x = 5
    #     int_x = ite_x * int_height
        while int_y + int_width * 3 <= img2.shape[1]:
        # if True:
        #     ite_y = 10
        #     int_y = ite_y * int_width
            # start5 = time.perf_counter()
            template = img2[int_x:int_x + int_height, int_y:int_y + int_width]

            method = eval('cv2.TM_CCORR_NORMED')
            # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            #             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

            scan_img = img1[int_x - int_height * 1:int_x + int_height * 2, int_y - int_width * 1:int_y + int_width * 2]
            # stop5 = time.perf_counter() - start5
            # print(f'stop5: {stop5}')


            # start3 = time.perf_counter()
            res = cv2.matchTemplate(scan_img, template, method)
            # stop3 = time.perf_counter() - start3
            # print(f'stop3: {stop3}')

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val < 0.95:
                part_img = color_img[int_x:int_x + int_height, int_y:int_y+int_width]
                white_pixels = (part_img == (255, 255, 255)).all(axis=2)
                part_img[white_pixels] = (255, 99, 71)
                color_img[int_x:int_x + int_height, int_y:int_y+int_width] = part_img

            #スライスして部分行列を取出し、その中の白い部分だけ赤くする
            # cv2.rectangle(color_img, top_left, bottom_right, (255, 99, 71), 2)

            ite_y += 1
            int_y = ite_y * int_width
        ite_y = 2
        int_y = ite_y * int_width
        ite_x += 1
        int_x = ite_x * int_height

    plt.subplot(141).imshow(res)
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(142).imshow(color_img)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.subplot(143)
    plt.imshow(template, vmin=0, vmax=255, cmap='gray')
    plt.subplot(144)
    plt.imshow(scan_img, vmin=0, vmax=255, cmap='gray')


    plt.show()

except:
    import sys
    print('Error: ', sys.exc_info()[0])
    print(sys.exc_info()[1])
    import traceback
    print(traceback.format_tb(sys.exc_info()[2]))


    # pdf_file_path = pathlib.Path('00.pdf')
    # base = pdf_file_path.stem

#    if img.shape[2] == 3:
#        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
#    for index, image in enumerate(images):
#        image.save(pathlib.Path(base + f'-{index + 1}.png'), 'png')


#    filename = "00-1.png"
#    img = cv2.imread(filename)

#    cv2.imshow('image',template)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

