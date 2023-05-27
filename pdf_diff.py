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
from sklearn.decomposition import PCA

# opencvとmatplotでx方向とy方向が異なる？
# opencvでは左上が原点でx方向が縦、y方向が横
# matplotでは左上が原点でx方向が横、y方向が縦

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

    # 検出される線分の長さは3000以上とする
    lines = cv2.HoughLinesP(reversed_img1, rho=1, theta=np.pi/180, threshold=50, minLineLength=1000, maxLineGap=10)
    img_disp = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    # 元画像上に線分を描画するための色
    red = (0, 0, 255)
    blue = (255, 0, 0)

    # 縦線と横線を格納するリスト
    v_lines = []
    h_lines = []

    # 角度によってフィルタリングする
    angle_threshold = np.pi / 4
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 線分の角度を計算する(angle=-1.54～+1.54, angle=0は横線)
        angle = np.arctan2(y2 - y1, x2 - x1)

        # 縦線か横線かを判定する
        if abs(abs(angle) - np.pi / 2) < angle_threshold:
            # 縦線の場合 → x1,x2がほぼ等しい
            v_lines.append(line)
            cv2.line(img_disp, (x1, y1), (x2, y2), red, 2)
        elif abs(angle) < angle_threshold:
            # 横線の場合 → y1,y2がほぼ等しい
            h_lines.append(line)
            cv2.line(img_disp, (x1, y1), (x2, y2), blue, 2)

    #場所によるフィルタリング
    img_disp = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    v_lines1 = []
    h_lines1 = []
    for line in h_lines:
        x1, y1, x2, y2 = line[0]
        if (y1 > 300 and y1 < 500) and (y2 > 300 and y2 < 500):
            h_lines1.append(line)
            cv2.line(img_disp, (x1, y1), (x2, y2), red, 2)
    
    for line in v_lines:
        x1, y1, x2, y2 = line[0]
        if x1 < 500 and x2 < 500:
            v_lines1.append(line)
            cv2.line(img_disp, (x1, y1), (x2, y2), blue, 2)
    
    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
    plt.imshow(img_disp)
    plt.show()
    print(f'検出数: 横線 {len(h_lines)}, 縦線 {len(v_lines)}')

    h_points = []

    for line in h_lines1:
        x1, y1, x2, y2 = line[0]
        h_points.append([x1, y1])
        h_points.append([x2, y2])

    pca = PCA(n_components=1)
    pca.fit(h_points)
    print(pca.components_)

    if pca.components_[0][1] < 0.1:
        y_values = [p[1] for p in h_points]
        y = sum(y_values) / len(y_values)

    v_points = []

    for line in v_lines1:
        x1, y1, x2, y2 = line[0]
        v_points.append([x1, y1])
        v_points.append([x2, y2])

    pca = PCA(n_components=1)
    pca.fit(v_points)
    print(pca.components_)

    if pca.components_[0][0] < 0.1:
        x_values = [p[0] for p in v_points]
        x = sum(x_values) / len(x_values)

    print("Intersection point: ({}, {})".format(int(x), int(y)))
    print('直線を描写しています')

    # 交点を整数値に変換
    intersection = (int(x), int(y))

    # 画像上に交点を表示する
    img_disp = cv2.line(img_disp, (intersection[0]-10, intersection[1]), (intersection[0]+10, intersection[1]), (0, 0, 255), 2)
    img_disp = cv2.line(img_disp, (intersection[0], intersection[1]-10), (intersection[0], intersection[1]+10), (0, 0, 255), 2)

    # print('比較元画像の基準点をクリックしてください')
    # fig = plt.figure()
    # plt.imshow(img1, vmin=0, vmax=255, cmap='gray')
    # fig.canvas.mpl_connect('button_press_event', onclick2)
    # plt.show()
    # if not posi:
    #     print('座標が指定されていません')
    #     sys.exit()
    # x1, y1 = posi.pop()

    # images will be a list of PIL Image representing each page of the PDF document.
    images2 = pdf2image.convert_from_path(settings['filename2'], grayscale=True, dpi=600)
    img2 = np.array(images2[0], dtype=np.uint8)
    print(f'比較先ファイル {settings["filename2"]} を読み込みました。画像サイズ: {img2.shape}')

    print('直線を検出しています')
    _, img2 = cv2.threshold(img2, 240, 255, cv2.THRESH_BINARY)
    reversed_img2 = cv2.bitwise_not(img2)

    # 検出される線分の長さは3000以上とする
    lines = cv2.HoughLinesP(reversed_img2, rho=1, theta=np.pi/180, threshold=50, minLineLength=1000, maxLineGap=10)
    img_disp = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # 縦線と横線を格納するリスト
    v_lines = []
    h_lines = []

    # 角度によってフィルタリングする
    angle_threshold = np.pi / 4
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 線分の角度を計算する(angle=-1.54～+1.54, angle=0は横線)
        angle = np.arctan2(y2 - y1, x2 - x1)

        # 縦線か横線かを判定する
        if abs(abs(angle) - np.pi / 2) < angle_threshold:
            # 縦線の場合 → x1,x2がほぼ等しい
            v_lines.append(line)
            cv2.line(img_disp, (x1, y1), (x2, y2), red, 2)
        elif abs(angle) < angle_threshold:
            # 横線の場合 → y1,y2がほぼ等しい
            h_lines.append(line)
            cv2.line(img_disp, (x1, y1), (x2, y2), blue, 2)

    #場所によるフィルタリング
    img_disp = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    v_lines1 = []
    h_lines1 = []
    for line in h_lines:
        x1, y1, x2, y2 = line[0]
        if (y1 > 300 and y1 < 500) and (y2 > 300 and y2 < 500):
            h_lines1.append(line)
            cv2.line(img_disp, (x1, y1), (x2, y2), red, 2)
    
    for line in v_lines:
        x1, y1, x2, y2 = line[0]
        if x1 < 500 and x2 < 500:
            v_lines1.append(line)
            cv2.line(img_disp, (x1, y1), (x2, y2), blue, 2)
    
    img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
    plt.imshow(img_disp)
    plt.show()
    print(f'検出数: 横線 {len(h_lines)}, 縦線 {len(v_lines)}')

    h_points = []

    for line in h_lines1:
        x1, y1, x2, y2 = line[0]
        h_points.append([x1, y1])
        h_points.append([x2, y2])

    pca = PCA(n_components=1)
    pca.fit(h_points)
    print(pca.components_)

    if pca.components_[0][1] < 0.1:
        y_values = [p[1] for p in h_points]
        yy = sum(y_values) / len(y_values)

    v_points = []

    for line in v_lines1:
        x1, y1, x2, y2 = line[0]
        v_points.append([x1, y1])
        v_points.append([x2, y2])

    pca = PCA(n_components=1)
    pca.fit(v_points)
    print(pca.components_)

    if pca.components_[0][0] < 0.1:
        x_values = [p[0] for p in v_points]
        xx = sum(x_values) / len(x_values)

    print("Intersection point: ({}, {})".format(int(xx), int(yy)))
    print('直線を描写しています')

    # 交点を整数値に変換
    intersection = (int(x), int(y))

    # 画像上に交点を表示する
    img_disp = cv2.line(img_disp, (intersection[0]-10, intersection[1]), (intersection[0]+10, intersection[1]), (0, 0, 255), 2)
    img_disp = cv2.line(img_disp, (intersection[0], intersection[1]-10), (intersection[0], intersection[1]+10), (0, 0, 255), 2)

    img1 = cv2.copyMakeBorder(img1, settings['border_x'], settings['border_x'], 
        settings['border_y'], settings['border_y'], cv2.BORDER_CONSTANT, value=255)
    print(f'比較元画像の周囲に余白を追加しました。画像サイズ: {img1.shape}')


    # 両者の原点位置により、オフセット量を計算し、比較元の画像img1をオフセットさせる
    delta_x = xx - x
    delta_y = yy - y
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


#最小二乗法で直線の方程式を求める
# https://dev.classmethod.jp/articles/pythonscikit-learn-pca1/
# https://qiita.com/supersaiakujin/items/138c0d8e6511735f1f45
#交点を求める
