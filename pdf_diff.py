import numpy as np
import cv2
import pdf2image
import pprint
import sys
import time
import traceback
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
            WHITE = (255, 255, 255)
            COLORED = (255, 99, 71)

            if max_val < settings['criterion']:
                part_img = cimg[x:x + settings['intr_area_x'], y:y + settings['intr_area_y']]
                white_pixels = (part_img == WHITE).all(axis=2)
                part_img[white_pixels] = COLORED
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

    color_img[:, :] = color_img2[:, :]
    shm.close()
    shm.unlink()


def get_origin(img):
    '''
    原点を求める

    自動で原点を検出し、画像を表示する。人間がそれを確認して気に入らなければ手動で原点を設定する
    '''
    global posi

    ANGLE_THRESHOLD = np.pi / 4
    MIN_LENGTH = 500
    MIN_X = 0
    MAX_X = 500
    MIN_Y = 300
    MAX_Y = 500

    RANGE_LIMIT = 50
    print(f'{MIN_LENGTH=}, {MIN_X=}, {MAX_X=}, {MIN_Y=}, {MAX_Y=}')

    print('直線を検出しています')
    _, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    reversed_img = cv2.bitwise_not(img)
    lines = cv2.HoughLinesP(reversed_img, rho=1, theta=np.pi/180, threshold=50, minLineLength=MIN_LENGTH, maxLineGap=10)
    # 確認用画像の準備
    img_disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print('直線をフィルタリングしています')

    # 角度によってフィルタリングする
    # 縦線横線の分類
    v_lines = []
    h_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 線分の角度を計算する(angle=-1.54～+1.54, angle=0は横線)
        angle = np.arctan2(y2 - y1, x2 - x1)

        # 縦線か横線かを判定する
        if abs(abs(angle) - np.pi / 2) < ANGLE_THRESHOLD:
            # 縦線の場合 → x1,x2がほぼ等しい
            v_lines.append(line)
        elif abs(angle) < ANGLE_THRESHOLD:
            # 横線の場合 → y1,y2がほぼ等しい
            h_lines.append(line)

    #場所によるフィルタリング
    h_lines1 = []
    for line in h_lines:
        x1, y1, x2, y2 = line[0]
        if (y1 > MIN_Y and y1 < MAX_Y) and (y2 > MIN_Y and y2 < MAX_Y):
            h_lines1.append(line)
            cv2.line(img_disp, (x1, y1), (x2, y2), settings['red'], 2)
    
    print(f'検出数: 横線 {len(h_lines1)}')

    h_points = []
    for line in h_lines1:
        x1, y1, x2, y2 = line[0]
        h_points.append([x1, y1])
        h_points.append([x2, y2])

    pca = PCA(n_components=1)
    pca.fit(h_points)
    print(pca.components_)

    if pca.components_[0][1] < 0.1:
        values = [p[1] for p in h_points]
        print(f'{values=}')

        if (max(values) - min(values)) > RANGE_LIMIT:
            hist, bins, _ = plt.hist(values)
            if settings['debug']:
                plt.show()

            highest_peak_bin = np.argmax(hist)
            start_bin = bins[highest_peak_bin]
            end_bin = bins[highest_peak_bin+1]
            peak_elements = [v for v in values if start_bin <= v < end_bin]
            print(f'{start_bin=}, {end_bin=}, {peak_elements=}')

            y_mean1 = sum(peak_elements) / len(peak_elements)
        else:
            y_mean1 = sum(values) / len(values)
    else:
        raise NotImplementedError()

    v_lines1 = []
    for line in v_lines:
        x1, y1, x2, y2 = line[0]
        if (x1 > MIN_X and x1 < MAX_X) and (x2 > MIN_X and x2 < MAX_X):
            v_lines1.append(line)
            cv2.line(img_disp, (x1, y1), (x2, y2), settings['blue'], 2)

    print(f'検出数: 縦線 {len(v_lines1)}')

    v_points = []
    for line in v_lines1:
        x1, y1, x2, y2 = line[0]
        v_points.append([x1, y1])
        v_points.append([x2, y2])

    pca = PCA(n_components=1)
    pca.fit(v_points)
    print(pca.components_)

    if pca.components_[0][0] < 0.1:
        values = [p[0] for p in v_points]
        print(f'{values=}')

        if (max(values) - min(values)) > RANGE_LIMIT:
            hist, bins, _ = plt.hist(values)
            if settings['debug']:
                plt.show()

            highest_peak_bin = np.argmax(hist)
            start_bin = bins[highest_peak_bin]
            end_bin = bins[highest_peak_bin+1]
            peak_elements = [v for v in values if start_bin <= v < end_bin]
            print(f'{start_bin=}, {end_bin=}, {peak_elements=}')

            x_mean1 = sum(peak_elements) / len(peak_elements)
        else:
            x_mean1 = sum(values) / len(values)
    else:
        raise NotImplementedError()

    # 交点を整数値に変換
    intersection = (int(x_mean1), int(y_mean1))
    print(f'{intersection=}')
 
    # 画像上に交点を表示する
    img_disp = cv2.line(img_disp, (intersection[0]-30, intersection[1]), (intersection[0]+30, intersection[1]), settings['green'], 5)
    img_disp = cv2.line(img_disp, (intersection[0], intersection[1]-30), (intersection[0], intersection[1]+30), settings['green'], 5)

    print('原点の位置が気に入らない場合には原点の位置をクリックしてください')

    posi = []
    fig = plt.figure()
    plt.imshow(img_disp)
    fig.canvas.mpl_connect('button_press_event', onclick2)
    plt.show()

    if posi:
        intersection = posi.pop()
        print(f'{intersection=}')

    return intersection

def main(settings):
    global posi

    # images will be a list of PIL Image representing each page of the PDF document.
    images1 = pdf2image.convert_from_path(settings['filename1'], grayscale=True, dpi=600)
    # 元バージョンでは1ページのみ処理する
    img1 = np.array(images1[0], dtype=np.uint8)
    img1_original = img1.copy()
    print(f'比較元ファイル {settings["filename1"]} を読み込みました。画像サイズ: {img1.shape}')

    x_mean1, y_mean1 = get_origin(img1)

    ###########################  比較先  #################################

    # images will be a list of PIL Image representing each page of the PDF document.
    images2 = pdf2image.convert_from_path(settings['filename2'], grayscale=True, dpi=600)
    img2 = np.array(images2[0], dtype=np.uint8)
    print(f'比較先ファイル {settings["filename2"]} を読み込みました。画像サイズ: {img2.shape}')

    x_mean2, y_mean2 = get_origin(img2)

    img1 = cv2.copyMakeBorder(img1, settings['border_x'], settings['border_x'], 
        settings['border_y'], settings['border_y'], cv2.BORDER_CONSTANT, value=255)
    print(f'比較元画像の周囲に余白を追加しました。画像サイズ: {img1.shape}')

    # 両者の原点位置により、オフセット量を計算し、比較元の画像img1をオフセットさせる
    delta_x = x_mean2 - x_mean1
    delta_y = y_mean2 - y_mean1
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

        if img2.shape[0] > img2.shape[1]:
            fig = plt.figure(figsize=(11.69, 8.27))
            
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.axis('off')
            ax1.set_position(mpl.transforms.Bbox([[0, 0], [0.5, 1]]))
            ax1.imshow(img1_original, vmin=0, vmax=255, cmap='gray')
            
            ax2 = plt.subplot(1, 2, 2)
            ax2.axis('off')
            ax2.set_position(mpl.transforms.Bbox([[0.5, 0], [1, 1]]))
            ax2.imshow(color_img)

            plt.savefig('yoko.pdf', dpi=600, bbox_inches='tight')
            fig.canvas.mpl_connect('button_press_event', onclick2)
            plt.show()
        else:
            fig = plt.figure(figsize=(8.27, 11.69))
            
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.axis('off')
            ax1.set_position(mpl.transforms.Bbox([[0, 0.5], [1, 1]]))
            ax1.imshow(img1_original, vmin=0, vmax=255, cmap='gray')
            
            ax2 = plt.subplot(2, 1, 2)
            ax2.axis('off')
            ax2.set_position(mpl.transforms.Bbox([[0, 0], [1, 0.5]]))
            ax2.imshow(color_img)

            plt.savefig('tate.pdf', dpi=600, bbox_inches='tight')
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
