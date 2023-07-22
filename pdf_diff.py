import numpy as np
import cv2
import pdf2image
import pprint
import sys
import time
import traceback
import matplotlib as mpl
import multiprocessing as mp
from concurrent.futures import (ProcessPoolExecutor, Future)
from functools import wraps
from matplotlib import pyplot as plt
from multiprocessing import Process, Queue
from multiprocessing import shared_memory
from merger import Merger
from presenter import ImagePresenterRaw, ImagePresenterInverseLeft, ImagePresenterInverseRight, ImagePresenterOverlap
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
        print(f'{f.__name__}の所要時間: {(time.time() - start):.2f} s')
        return v
    return wrapper

def onclick2(event):
    global posi

    # matplotlibでは横方向がx、縦方向がyになる模様
    posi.append([int(event.ydata), int(event.xdata)])
    print(f'Clicked point: {posi[-1]}')

def work(settings, img1, img2, similarity, shmname, process_num, start, stop=None):
    print(f'{shmname}-{process_num} starting')
    if stop is None:
        stop = int(img2.shape[0] / (settings['intr_area_x'] * settings['step_x']))

    existing_shm = shared_memory.SharedMemory(name=shmname)
    cimg = np.ndarray(similarity.shape, dtype=np.float32, buffer=existing_shm.buf)

    ite_xs = range(start, stop)
    for ite_x in ite_xs:
        x = int(ite_x * settings['intr_area_x'] * settings['step_x'])

        for ite_y in settings['ite_ys']:
            y = int(ite_y * settings['intr_area_y'] * settings['step_y'])

            # print(f'{ite_y=}, {y=}, {img2.shape[1]=}')
            template = img2[x:x + settings['intr_area_x'], y:y + settings['intr_area_y']]

            method = eval('cv2.TM_CCORR_NORMED')

            scan_img = img1[settings['border_x'] + x - int(settings['intr_area_x'] * settings['scan_area_ratio_x'])
                :settings['border_x'] + x + int(settings['intr_area_x'] * (settings['scan_area_ratio_x'] + 1)),
                settings['border_y'] + y - int(settings['intr_area_y'] * settings['scan_area_ratio_y'])
                :settings['border_y'] + y + int(settings['intr_area_y'] * (settings['scan_area_ratio_y'] + 1))]

            res = cv2.matchTemplate(scan_img, template, method)

            _, max_val, _, _ = cv2.minMaxLoc(res)

            cimg[x:x + settings['intr_area_x'], y:y + settings['intr_area_y']] += max_val
    existing_shm.close()
    print(f'{shmname}-{process_num} completed')

@elapsed_time
def get_similarity(queue, settings, img1, img2, similarity, shmname):
    settings['ite_ys'] = range(int((img2.shape[1] - settings['intr_area_y'])
        / (settings['intr_area_y'] * settings['step_y']) + 1))

    # 共有メモリの作成
    shm = shared_memory.SharedMemory(name=shmname, create=True, size=similarity.nbytes)

    # 共有メモリによって使用させるndarray
    similarity2 = np.ndarray(similarity.shape, dtype=np.float32, buffer=shm.buf)
    similarity2[:, :] = similarity[:, :]

    max_ite_x = int(img2.shape[0] / (settings['intr_area_x'] * settings['step_x'])) - 1

    num_process = 2
    processes = []
    for i in range(num_process):
        p = Process(target=work, args=(settings, img1, img2, similarity, shmname, i,
                                       int(max_ite_x / num_process * i), int(max_ite_x / num_process * (i + 1))))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    similarity[:, :] = similarity2[:, :]

    queue.put(similarity)
    shm.close()
    shm.unlink()
    if shmname == 'similarity_shm1':
        print('逆方向比較プロセス完了')
    elif shmname == 'similarity_shm2':
        print('正方向比較プロセス完了')
    else:
        raise NotImplementedError

def get_origin(img):
    """原点を求める

    自動で原点を検出し、画像を表示する。人間がそれを確認して適切でなければ手動で原点を設定する
    """
    global posi

    ANGLE_THRESHOLD = np.pi / 4
    MIN_LENGTH = 500
    MIN_X = 0
    MAX_X = 500
    MIN_Y = 300
    MAX_Y = 500

    RANGE_LIMIT = 50

    if settings['debug']:
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

    try:
        #場所によるフィルタリング
        h_lines1 = []
        for line in h_lines:
            x1, y1, x2, y2 = line[0]
            if (y1 > MIN_Y and y1 < MAX_Y) and (y2 > MIN_Y and y2 < MAX_Y):
                h_lines1.append(line)
                cv2.line(img_disp, (x1, y1), (x2, y2), settings['cv_red'], 2)
        
        print(f'横線(y方向)検出数: {len(h_lines1)}')

        if not len(h_lines1):
            raise IndexError
        
        h_points = []
        for line in h_lines1:
            x1, y1, x2, y2 = line[0]
            h_points.append([x1, y1])
            h_points.append([x2, y2])

        pca = PCA(n_components=1)
        pca.fit(h_points)
        if settings['debug']:
            print(f'{pca.components_=}')

        if pca.components_[0][1] < 0.1:
            values = [p[1] for p in h_points]
            if settings['debug']:
                print(f'{values=}')

            if (max(values) - min(values)) > RANGE_LIMIT:
                hist, bins, _ = plt.hist(values)
                if settings['debug']:
                    plt.show()
                else:
                    plt.close()

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

    except IndexError:
        y_mean1 = 0
    
    except NotImplementedError:
        y_mean1 = 0

    try:
        v_lines1 = []
        for line in v_lines:
            x1, y1, x2, y2 = line[0]
            if (x1 > MIN_X and x1 < MAX_X) and (x2 > MIN_X and x2 < MAX_X):
                v_lines1.append(line)
                cv2.line(img_disp, (x1, y1), (x2, y2), settings['cv_blue'], 2)

        print(f'縦線(x方向)検出数: {len(v_lines1)}')

        if not len(v_lines1):
            raise IndexError
        
        v_points = []
        for line in v_lines1:
            x1, y1, x2, y2 = line[0]
            v_points.append([x1, y1])
            v_points.append([x2, y2])

        pca = PCA(n_components=1)
        pca.fit(v_points)
        if settings['debug']:
            print(f'{pca.components_=}')

        if pca.components_[0][0] < 0.1:
            values = [p[0] for p in v_points]
            if settings['debug']:
                print(f'{values=}')

            if (max(values) - min(values)) > RANGE_LIMIT:
                hist, bins, _ = plt.hist(values)
                if settings['debug']:
                    plt.show()
                else:
                    plt.close()

                highest_peak_bin = np.argmax(hist)
                start_bin = bins[highest_peak_bin]
                end_bin = bins[highest_peak_bin+1]
                peak_elements = [v for v in values if start_bin <= v < end_bin]
                print(f'{start_bin=}, {end_bin=}, {peak_elements=}')

                x_mean1 = sum(peak_elements) / len(peak_elements)
            else:
                x_mean1 = sum(values) / len(values)
        else:
            raise NotImplementedError
        
    except IndexError:
        x_mean1 = 0
    
    except NotImplementedError:
        x_mean1 = 0

    # 交点を整数値に変換
    intersection = (int(x_mean1), int(y_mean1))
    print(f'原点: {intersection}')
 
    if settings['check_origin']:
        # 画像上に交点を表示する
        img_disp = cv2.line(img_disp, (intersection[0]-30, intersection[1]), (intersection[0]+30, intersection[1]), settings['cv_green'], 5)
        img_disp = cv2.line(img_disp, (intersection[0], intersection[1]-30), (intersection[0], intersection[1]+30), settings['cv_green'], 5)

        print('原点の位置が適正でない場合には原点の位置をクリックしてください')

        posi = []
        fig = plt.figure()
        plt.imshow(img_disp)
        fig.canvas.mpl_connect('button_press_event', onclick2)
        plt.show()

        if posi:
            intersection = posi.pop()
            print(f'原点位置を変更します: {intersection}')
    else:
        print('設定値check_originがFalseのため、原点の確認をスキップします')

    return intersection

def compare(settings, images1, images2, page_num = 0):
    """各ページごとに比較を行い、結果を保存する
    """
    if page_num == settings['total_images1']:
        return
    
    print(f'ページ: {page_num}')
    print(f'============= STEP 02 - 01 ({page_num}ページ目) =============')
    img1 = np.array(images1[page_num], dtype=np.uint8)

    if settings['rotate1'] == 'cw':
        img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
    if settings['rotate1'] == 'ccw':
        img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

    print(f'比較元画像 {settings["filename1"]} を読み込みました。画像サイズ: {img1.shape}')

    x_mean1, y_mean1 = get_origin(img1)

    print(f'\n============= STEP 02 - 02 ({page_num}ページ目) =============')
    img2 = np.array(images2[page_num], dtype=np.uint8)

    if settings['rotate2'] == 'cw':
        img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    if settings['rotate1'] == 'ccw':
        img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    print(f'比較先ファイル {settings["filename2"]} を読み込みました。画像サイズ: {img2.shape}')

    x_mean2, y_mean2 = get_origin(img2)

    print(f'\n============= STEP 02 - 03 ({page_num}ページ目) =============')
    print('比較元ファイルの位置合わせを行います')

    print(f'必要な片側余白量(x方向): {settings["border_x"]}')
    print(f'必要な片側余白量(y方向): {settings["border_y"]}')
    img1_margined = cv2.copyMakeBorder(img1, settings['border_x'], settings['border_x'], 
        settings['border_y'], settings['border_y'], cv2.BORDER_CONSTANT, value=255)
    print(f'比較元画像の周囲に余白を追加しました。画像サイズ: {img1_margined.shape}')

    # 両者の原点位置により、オフセット量を計算し、比較元の画像img1をオフセットさせる
    settings['delta_x'] = x_mean2 - x_mean1
    settings['delta_y'] = y_mean2 - y_mean1
    print('比較元画像の位置を調整します')
    print(f'x方向移動量: {settings["delta_x"]} (下が正方向)')
    print(f'y方向移動量: {settings["delta_y"]} (右が正方向)')

    M = np.float32([[1, 0, settings['delta_x']], [0, 1, settings['delta_y']]])
    img1_margined = cv2.warpAffine(img1_margined, M, (img1_margined.shape[1], img1_margined.shape[0]), borderValue=255)
    print(f'比較元画像の位置調整を行いました')

    print(f'\n============= STEP 02 - 04 ({page_num}ページ目) =============')
    print('比較先ファイルの位置合わせを行います')

    print(f'必要な片側余白量(x方向): {settings["border_x"]}')
    print(f'必要な片側余白量(y方向): {settings["border_y"]}')
    img2_margined = cv2.copyMakeBorder(img2, settings['border_x'], settings['border_x'], 
        settings['border_y'], settings['border_y'], cv2.BORDER_CONSTANT, value=255)
    print(f'比較先画像の周囲に余白を追加しました。画像サイズ: {img2_margined.shape}')

    # 比較先の画像img2をオフセットさせる
    print('比較先画像の位置を調整します')
    print(f'x方向移動量: {-settings["delta_x"]} (下が正方向)')
    print(f'y方向移動量: {-settings["delta_y"]} (右が正方向)')

    M = np.float32([[1, 0, -settings['delta_x']], [0, 1, -settings['delta_y']]])
    img2_margined = cv2.warpAffine(img2_margined, M, (img2_margined.shape[1], img2_margined.shape[0]), borderValue=255)
    print(f'比較先画像の位置調整を行いました')

    print(f'\n============= STEP 02 - 05 ({page_num}ページ目) =============')
    if settings['display'] == 'comparison':
        print('類似度を計算します')

        result_queue1 = Queue()
        result_queue2 = Queue()

        similarity2 = np.zeros_like(img2, dtype=np.float32)
        similarity1 = np.zeros_like(img1, dtype=np.float32)

        print('比較元から比較先への類似度の計算を行います(正方向比較プロセス)')
        p1 = Process(target=get_similarity, args=(result_queue2, settings, img1_margined, img2, similarity2, 'similarity_shm2'))
        p1.start()
        print('比較先から比較元への類似度の計算を行います(逆方向比較プロセス)')
        p2 = Process(target=get_similarity, args=(result_queue1, settings, img2_margined, img1, similarity1, 'similarity_shm1'))
        p2.start()
        similarity2 = result_queue2.get()
        similarity1 = result_queue1.get()

        p1.join()
        p2.join()
        print('完了')

        if settings['check_similarity']:
            print('類似度(similarity2)をCSVファイルに出力しています...', end=' ')
            np.savetxt('similarity2.csv', similarity2, delimiter=',', fmt='%3.3f')
            print('完了')
    else:
        print('類似度の計算をスキップします')

    print(f'\n============= STEP 02 - 06 ({page_num}ページ目) =============')
    print('結果を表示します')

    if settings['display'] == 'raw':
        ip = ImagePresenterRaw(settings, page_num, img1, img2)
    elif settings['display'] == 'overlap':
        ip = ImagePresenterOverlap(settings, page_num, img1, img2)
    elif settings['display'] == 'comparison':
        if settings['inverse_comparison'] == 'left':
            ip = ImagePresenterInverseLeft(settings, page_num, img1, img2, similarity1, similarity2)
        elif settings['inverse_comparison'] == 'right':
            ip = ImagePresenterInverseRight(settings, page_num, img1, img2, similarity1, similarity2)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    ip.show()

    compare(settings, images1, images2, page_num + 1)


def main(settings):
    '''
    PDFを読み込み比較する
    '''
    global posi

    print('\n============= STEP 01 =============')
    print('ファイルの読み込みを行います')
    print(f'比較元ファイル: {settings["filename1"]}', end=' ')
    # images will be a list of PIL Image representing each page of the PDF document.
    images1 = pdf2image.convert_from_path(settings['filename1'], grayscale=True, dpi=600)
    print('完了')

    settings['total_images1'] = len(images1)
    print(f'ページ数: {settings["total_images1"]}')

    print(f'比較先ファイル: {settings["filename2"]}', end=' ')
    images2 = pdf2image.convert_from_path(settings['filename2'], grayscale=True, dpi=600)
    print('完了')

    settings['total_images2'] = len(images2)
    print(f'ページ数: {settings["total_images2"]}')

    try:
        print('\n============= STEP 02 =============')
        compare(settings, images1, images2)

        print('\n============= STEP 03 =============')
        print('ファイルを結合します')
        merger = Merger(settings)
        merger.execute()
    except PermissionError:
        print('ファイルの書き込み時に問題が発生しました')
        print('ファイルが開いたままではないか確認してください')
        return

if __name__ == '__main__':
    print('============= STEP 00 =============')
    print('比較用のファイル名を取得しています')
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
