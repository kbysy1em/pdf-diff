import numpy as np
import cv2
import pdf2image
import sys
import time
import traceback
import matplotlib as mpl
from matplotlib import pyplot as plt

posi = []

def onclick2(event):
    global posi

    # matplotlibでは横方向がx、縦方向がyになる模様
    posi.append([int(event.ydata), int(event.xdata)])
    print(f'Clicked point: {posi[-1]}')


def main(input_filename1, input_filename2):
    global posi

    # images will be a list of PIL Image representing each page of the PDF document.
    images1 = pdf2image.convert_from_path(input_filename1, grayscale=True, dpi=600)
    img1 = np.array(images1[0], dtype=np.uint8)
    print(f'比較元ファイル {input_filename1} を読み込みました。画像サイズ: {img1.shape}')

    print('比較元の原点をクリックしてください')
    fig = plt.figure()
    plt.imshow(img1, vmin=0, vmax=255, cmap='gray')
    fig.canvas.mpl_connect('button_press_event', onclick2)
    plt.show()
    x1, y1 = posi.pop()

    images2 = pdf2image.convert_from_path(input_filename2, grayscale=True, dpi=600)
    img2 = np.array(images2[0], dtype=np.uint8)
    print(f'比較先ファイル {input_filename2} を読み込みました。画像サイズ: {img2.shape}')

    print('比較先の原点をクリックしてください')
    fig = plt.figure()
    plt.imshow(img2, vmin=0, vmax=255, cmap='gray')
    fig.canvas.mpl_connect('button_press_event', onclick2)
    plt.show()
    x2, y2 = posi.pop()

    # 両者の原点位置により、オフセット量を計算し、比較元の画像img1をオフセットさせる
    delta_x = x2 - x1
    delta_y = y2 - y1
    print(f'{x1=}, {y1=}, {x2=}, {y2=}, {delta_x=}, {delta_y=}')

    M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    img1 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]), borderValue=255)

    color_img = cv2.merge((img2, img2, img2))

    # 縦がX(下向きが正)、横がY(右向きが正)
    # イントロゲーションエリアの原点は左上 
    int_height = 100
    int_width = 100
    step_x = 0.5
    step_y = 0.5
    scan_area_ratio_x = 0.5
    scan_area_ratio_y = 0.5
    initial_ite_x = int(scan_area_ratio_x / step_x)  # 初期値
    initial_ite_y = int(scan_area_ratio_y / step_y)  # 初期値

    ite_xs = range(initial_ite_x, 800)
    ite_ys = range(initial_ite_y, 200)

    start1 = time.perf_counter()
    # while current_x() + int_height * (scan_area_ratio_x + 1) < img2.shape[0]:
    for ite_x in ite_xs:
        x = int(ite_x * int_height * step_x)

        for ite_y in ite_ys:
            y = int(ite_y * int_width * step_y)

            # print(f'{ite_y=}, {y=}, {img2.shape[1]=}')
            template = img2[x:x + int_height, y:y + int_width]

            method = eval('cv2.TM_CCORR_NORMED')
            # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            #             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

            scan_img = img1[x - int(int_height * scan_area_ratio_x)
                :x + int(int_height * (scan_area_ratio_x + 1)),
                y - int(int_width * scan_area_ratio_y)
                :y + int(int_width * (scan_area_ratio_y + 1))]

            res = cv2.matchTemplate(scan_img, template, method)

            _, max_val, _, _ = cv2.minMaxLoc(res)

            # 一致度が低い場所は色を付ける
            if max_val < 0.95:
                part_img = color_img[x:x + int_height, y:y + int_width]
                white_pixels = (part_img == (255, 255, 255)).all(axis=2)
                part_img[white_pixels] = (255, 99, 71)
                color_img[x:x + int_height, y:y + int_width] = part_img

            if y + int_width + scan_area_ratio_y * int_width > img2.shape[1]:
                break
            
        if x + int_height + scan_area_ratio_x * int_height > img2.shape[0]:
            break

    print(f'stop_all: {((time.perf_counter()-start1)*10**6):.1f}')
#5157198 while
#5102536 for loop
    retry = True


    while retry:
        posi = []

        print('一部の着色箇所を白に戻すには、戻したい箇所(長方形2点)を選択した上で、画像を閉じ\n'
            + '次に現れるダイアログで n を入力してください')

        fig = plt.figure(figsize=(11.69, 8.27))
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.axis('off')
        ax1.set_position(mpl.transforms.Bbox([[0, 0], [0.5, 1]]))
        ax1.imshow(img1, vmin=0, vmax=255, cmap='gray')
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.axis('off')
        ax2.set_position(mpl.transforms.Bbox([[0.5, 0], [1, 1]]))
        ax2.imshow(color_img)

        plt.savefig('aaa.pdf', dpi=600, bbox_inches='tight')
        fig.canvas.mpl_connect('button_press_event', onclick2)
        plt.show()

        if len(posi) < 2:
            break

        if input('終了してよろしいですか?(y/n)') != 'n':
            break

        x2, y2 = posi.pop()
        x1, y1 = posi.pop()

        print(x1, x2, y1, y2)
        part_img = color_img[x1:x2, y1:y2]

        red_pixels = (part_img == (255, 99, 71)).all(axis=2)
        part_img[red_pixels] = (255, 255, 255)
        color_img[x1:x2, y1:y2] = part_img

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise Exception('引数が足りません')

        main(sys.argv[1], sys.argv[2])

    except:
        print('Error: ', sys.exc_info()[0])
        print(sys.exc_info()[1])
        print(traceback.format_tb(sys.exc_info()[2]))
