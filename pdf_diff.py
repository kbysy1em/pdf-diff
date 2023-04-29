import numpy as np
import cv2
import csv
import pdf2image
import pprint
import random
import sys
import time
import traceback
import math
import matplotlib as mpl
from concurrent.futures import (ProcessPoolExecutor, Future)
from functools import wraps
from matplotlib import pyplot as plt
from multiprocessing import Process
from multiprocessing import shared_memory
from settings import *

posi = []

def elapsed_time(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        v = f(*args, **kwargs)
        print(f"{f.__name__}: {time.time() - start}")
        return v
    return wrapper

def onclick2(event):
    global posi

    # matplotlibでは横方向がx、縦方向がyになる模様
    posi.append([int(event.ydata), int(event.xdata)])
    print(f'Clicked point: {posi[-1]}')

def work(settings, img1, img2, color_img, start, stop=None):
    if stop is None:
        stop = int((img2.shape[0] - settings['intr_area_x']) / (settings['intr_area_x'] * settings['step_x']))

    cshm = shared_memory.SharedMemory(name='color_img_shm')
    cimg = np.ndarray(color_img.shape, dtype=np.uint8, buffer=cshm.buf)

    ite_xs = range(start, stop)

    for ite_x in ite_xs:
        x = int(ite_x * settings['intr_area_x'] * settings['step_x'])

        for ite_y in settings['ite_ys']:
            y = int(ite_y * settings['intr_area_y'] * settings['step_y'])

            # print(f'{ite_y=}, {y=}, {img2.shape[1]=}')
            template = img2[x:x + settings['intr_area_x'], y:y + settings['intr_area_y']]

            method = eval('cv2.TM_CCORR_NORMED')
            # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            #             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

            scan_img = img1[settings['border_x'] + x - int(settings['intr_area_x'] * settings['scan_area_ratio_x'])
                :settings['border_x'] + x + int(settings['intr_area_x'] * (settings['scan_area_ratio_x'] + 1)),
                settings['border_y'] + y - int(settings['intr_area_y'] * settings['scan_area_ratio_y'])
                :settings['border_y'] + y + int(settings['intr_area_y'] * (settings['scan_area_ratio_y'] + 1))]

            res = cv2.matchTemplate(scan_img, template, method)

            _, max_val, _, _ = cv2.minMaxLoc(res)

            # 一致度が低い場所は色を付ける
            if max_val < 0.95:
                part_img = cimg[x:x + settings['intr_area_x'], y:y + settings['intr_area_y']]
                white_pixels = (part_img == (255, 255, 255)).all(axis=2)
                part_img[white_pixels] = (255, 99, 71)
                cimg[x:x + settings['intr_area_x'], y:y + settings['intr_area_y']] = part_img
    cshm.close()

@elapsed_time
def check_and_color_img(settings, img1, img2, color_img):
    # settings['ite_xs'] = range(int((img2.shape[0] - settings['intr_area_x']) 
    #     / (settings['intr_area_x'] * settings['step_x'])))
    settings['ite_ys'] = range(int((img2.shape[1] - settings['intr_area_y'])
        / (settings['intr_area_y'] * settings['step_y'])))

    # print(color_img.dtype)
    shm = shared_memory.SharedMemory(name='color_img_shm', create=True, size=color_img.nbytes)

    color_img2 = np.ndarray(color_img.shape, dtype=np.uint8, buffer=shm.buf)
    color_img2[:, :] = color_img[:, :]

    p1 = Process(target=work, args=(settings, img1, img2, color_img, 0, 50))
    p2 = Process(target=work, args=(settings, img1, img2, color_img, 50, 100))
    p3 = Process(target=work, args=(settings, img1, img2, color_img, 100))

    p1.start()
    p2.start()
    p3.start()
    
    p1.join()
    p2.join()
    p3.join()

    plt.figure()
    plt.imshow(color_img2, vmin=0, vmax=255, cmap='gray')
    plt.show()

    color_img[:, :] = color_img2[:, :]
    shm.close()
    shm.unlink()


def main(settings):
    global posi

    # images will be a list of PIL Image representing each page of the PDF document.
    images1 = pdf2image.convert_from_path(settings['filename1'], grayscale=True, dpi=600)
    img1 = np.array(images1[0], dtype=np.uint8)
    print(f'比較元ファイル {settings["filename1"]} を読み込みました。画像サイズ: {img1.shape}')


    print('直線を検出しています')
    _, img1 = cv2.threshold(img1, 240, 255, cv2.THRESH_BINARY)
    reversed_img1 = cv2.bitwise_not(img1)
    # plt.imshow(reversed_img1, cmap='gray')
    # plt.show()

    lines = cv2.HoughLinesP(reversed_img1, rho=1, theta=np.pi/360, threshold=100, minLineLength=1000, maxLineGap=10)
    img_disp = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    
    ## 画像の左半分の中で最も長い縦線を抽出する
    # lines = map(lambda x: x.append(1), lines )
    # 縦線でかつ画像の右側にある最も長い線分(line_a)を求める
    line_a = None
    max_length_a = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if (x1 < img_disp.shape[1]/2) and (x2 < img_disp.shape[1]/2) and (min(y1, y2) < 500) and (length > max_length_a) and abs(y2-y1) > abs(x2-x1):
            max_length_a = length
            line_a = line

    # 横線でかつ画像の上側にある最も長い線分(line_b)を求める
    line_b = None
    max_length_b = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if (y1 < img_disp.shape[0]/2) and (y2 < img_disp.shape[0]/2) and (length > max_length_b) and abs(x2-x1) > abs(y2-y1):
            max_length_b = length
            line_b = line

    # line_aとline_bの交点を求める
    x1, y1, x2, y2 = line_a[0]
    x3, y3, x4, y4 = line_b[0]

    print(line_a)
    print(line_b)
    x = int((x1 + x2) / 2)
    y = int((y3 + y4) / 2)

    # denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    # print(denom)
    # if denom != 0:
    #    x = np.int64(((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / denom)
    #    y = np.int64(((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / denom)
    #     print((np.float64(x1)*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))
    #     x = (np.float64(x1)*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) / np.float64(denom)
    #     x = int(round(x))

    #     y = (np.float64(x1)*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) / np.float64(denom)
    #     y = int(round(y))
    #     print("Intersection point: ({}, {})".format(int(x), int(y)))
    # else:
    #     print("Lines are parallel")

    print("Intersection point: ({}, {})".format(int(x), int(y)))
    print('直線を描写しています')
    # for line in lines:
    #     # 原点は左上で縦方向がY
    #     x1, y1, x2, y2 = line[0]

    #     # 赤線を引く
    #     b = random.randrange(0, 255, 1)
    #     g = random.randrange(0, 255, 1)
    #     r = random.randrange(0, 255, 1)
    #     img_disp = cv2.line(img_disp, (x1,y1), (x2,y2), (b,g,r), 1)
    # 原点は左上で縦方向がY
    print(line_a[0])
    x1, y1, x2, y2 = line_a[0]
    # 赤線を引く
    b = random.randrange(0, 255, 1)
    g = random.randrange(0, 255, 1)
    r = random.randrange(0, 255, 1)
    img_disp = cv2.line(img_disp, (x1,y1), (x2,y2), (b,g,r), 3)

    x1, y1, x2, y2 = line_b[0]
    # 赤線を引く
    b = random.randrange(0, 255, 1)
    g = random.randrange(0, 255, 1)
    r = random.randrange(0, 255, 1)
    img_disp = cv2.line(img_disp, (x1,y1), (x2,y2), (b,g,r), 3)

    # 交点を整数値に変換
    intersection = (int(x), int(y))
    print(intersection)
    # 画像上に交点を表示する
    img_disp = cv2.line(img_disp, (intersection[0]-10, intersection[1]), (intersection[0]+10, intersection[1]), (0, 0, 255), 2)
    img_disp = cv2.line(img_disp, (intersection[0], intersection[1]-10), (intersection[0], intersection[1]+10), (0, 0, 255), 2)


    # rects = []
    # # 輪郭の点の描画
    # for contour in contours:
    #     # 傾いた外接する矩形領域
    #     rect = cv2.minAreaRect(contour)
    #     rects.append(rect)

    # rects.sort(key=lambda x: x[1][0]**2 + x[1][1]**2, reverse=True)

    # rect = rects[1]
    # box = cv2.boxPoints(rect)
    # box = np.intp(box)
    # print(box)
    # cv2.drawContours(img_disp, [box], 0, (0, 0, 255), 2)

    # points1 = sorted(box, key=lambda x: x[0]+x[1])
    # y1 = points1[0][0]
    # x1 = points1[0][1]
    # print(f'{x1=}, {y1=}')

    plt.imshow(img_disp)
    plt.show()

    # print('比較元画像の基準点をクリックしてください')
    # fig = plt.figure()
    # plt.imshow(img1, vmin=0, vmax=255, cmap='gray')
    # fig.canvas.mpl_connect('button_press_event', onclick2)
    # plt.show()
    # if not posi:
    #     print('座標が指定されていません')
    #     sys.exit()
    # x1, y1 = posi.pop()

    images2 = pdf2image.convert_from_path(settings['filename2'], grayscale=True, dpi=600)
    img2 = np.array(images2[0], dtype=np.uint8)
    print(f'比較先ファイル {settings["filename2"]} を読み込みました。画像サイズ: {img2.shape}')

    print('直線を検出しています')
    _, img2 = cv2.threshold(img2, 240, 255, cv2.THRESH_BINARY)
    reversed_img2 = cv2.bitwise_not(img2)

    lines = cv2.HoughLinesP(reversed_img2, rho=1, theta=np.pi/360, threshold=100, minLineLength=1000, maxLineGap=10)
    img_disp = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    print('直線を描写しています')
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 赤線を引く
        img_disp = cv2.line(img_disp, (x1,y1), (x2,y2), (0,0,255), 3)
    plt.imshow(img_disp)
    plt.show()
    sys.exit()

    print('輪郭を検出しています')
    _, img2 = cv2.threshold(img2, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    img_disp = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    rects = []
    # 輪郭の点の描画
    for i, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        rects.append(rect)

    rects.sort(key=lambda x: x[1][0]**2 + x[1][1]**2, reverse=True)


    rect = rects[1]
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    print(box)
    cv2.drawContours(img_disp, [box], 0, (0, 0, 255), 2)

    points2 = sorted(box, key=lambda x: x[0]+x[1])
    y2 = points2[0][0]
    x2 = points2[0][1]
    print(f'{x2=}, {y2=}')

    plt.imshow(img_disp)
    plt.show()

    # print('比較先画像の基準点をクリックしてください')
    # fig = plt.figure()
    # plt.imshow(img2, vmin=0, vmax=255, cmap='gray')
    # fig.canvas.mpl_connect('button_press_event', onclick2)
    # plt.show()
    # if not posi:
    #     print('座標が指定されていません')
    #     sys.exit()
    # x2, y2 = posi.pop()

    img1 = cv2.copyMakeBorder(img1, settings['border_x'], settings['border_x'], 
        settings['border_y'], settings['border_y'], cv2.BORDER_CONSTANT, value=255)
    print(f'比較元画像の周囲に余白を追加しました。画像サイズ: {img1.shape}')

    # 両者の原点位置により、オフセット量を計算し、比較元の画像img1をオフセットさせる
    delta_x = x2 - x1
    delta_y = y2 - y1
    print(f'移動量: {delta_x=}, {delta_y=}')

    M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    img1 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]), borderValue=255)
    print(f'比較元画像の位置調整を行いました')

    color_img = cv2.merge((img2, img2, img2))

    check_and_color_img(settings, img1, img2, color_img)

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

        settings['filename1'] = sys.argv[1]
        settings['filename2'] = sys.argv[2]
        main(settings)

    except SystemExit as e:
        print('終了します')
        print(e)

    except:
        print('Error: ')
        pprint.pprint(sys.exc_info())
        pprint.pprint(traceback.format_tb(sys.exc_info()[2]))
        # pprint.pprint('Error: ', sys.exc_info()[0])
        # pprint.pprint(sys.exc_info()[1])
        # pprint.pprint(traceback.format_tb(sys.exc_info()[2]))
