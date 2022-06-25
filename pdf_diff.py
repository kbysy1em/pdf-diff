import numpy as np
import cv2
import pdf2image
import sys
import time
from matplotlib import pyplot as plt

posi = []

class PointList():
    def __init__(self, npoints):
        self.npoints = npoints
        self.ptlist = np.empty((npoints, 2), dtype=int)
        self.pos = 0

    def add(self, x, y):
        if self.pos < self.npoints:
            self.ptlist[self.pos, :] = [x, y]
            self.pos += 1
            return True
        return False


def onMouse(event, x, y, flag, params):
    wname, img, ptlist = params
    if event == cv2.EVENT_MOUSEMOVE:  # マウスが移動したときにx線とy線を更新する
        img2 = np.copy(img)
        h, w = img2.shape[0], img2.shape[1]
        cv2.line(img2, (x, 0), (x, h - 1), (50, 205, 50))
        cv2.line(img2, (0, y), (w - 1, y), (50, 205, 50))
        cv2.imshow(wname, img2)

    if event == cv2.EVENT_LBUTTONDOWN:  # レフトボタンをクリックしたとき、ptlist配列にx,y座標を格納する
        if ptlist.add(x, y):
            print('[%d] ( %d, %d )' % (ptlist.pos - 1, x, y))
            cv2.circle(img, (x, y), 3, (50, 205, 50), 3)
            cv2.imshow(wname, img)
        else:
            print('All points have selected.  Press ESC-key.')
        if(ptlist.pos == ptlist.npoints):
            print(ptlist.ptlist)
            cv2.line(img, (ptlist.ptlist[0][0], ptlist.ptlist[0][1]),
                     (ptlist.ptlist[1][0], ptlist.ptlist[1][1]), (50, 205, 50), 3)
            cv2.line(img, (ptlist.ptlist[1][0], ptlist.ptlist[1][1]),
                     (ptlist.ptlist[2][0], ptlist.ptlist[2][1]), (50, 205, 50), 3)
            cv2.line(img, (ptlist.ptlist[2][0], ptlist.ptlist[2][1]),
                     (ptlist.ptlist[3][0], ptlist.ptlist[3][1]), (50, 205, 50), 3)
            cv2.line(img, (ptlist.ptlist[3][0], ptlist.ptlist[3][1]),
                     (ptlist.ptlist[0][0], ptlist.ptlist[0][1]), (50, 205, 50), 3)

def onclick2(event):
    global posi
    posi.append([event.xdata, event.ydata])

def main(input_filename1, input_filename2):
    global posi

    # images will be a list of PIL Image representing each page of the PDF document.
    images1 = pdf2image.convert_from_path(input_filename1, grayscale=True, dpi=600)
    img1 = np.array(images1[0], dtype=np.uint8)
    npoints = 4
    ptlist = PointList(npoints)

    fig = plt.figure()
    plt.imshow(img1)
    fig.canvas.mpl_connect('button_press_event', onclick2)
    plt.show()
    print(posi)

    x1, y1 = posi.pop()

    # cv2.namedWindow(wname)
    # cv2.imshow(wname, img1)
    # cv2.setMouseCallback(wname, onMouse, [wname, img1, ptlist])
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    

    images2 = pdf2image.convert_from_path(input_filename2, grayscale=True, dpi=600)
    img2 = np.array(images2[0], dtype=np.uint8)

    fig = plt.figure()
    plt.imshow(img2)
    fig.canvas.mpl_connect('button_press_event', onclick2)
    plt.show()
    print(posi)

    x2, y2 = posi.pop()

    # wname2 = '222'
    # cv2.namedWindow(wname2)
    # cv2.imshow(wname2, img2)
    # cv2.setMouseCallback(wname2, onMouse, [wname2, img2, ptlist])
    # cv2.waitKey()
    # cv2.destroyAllWindows()


    delta_x = x2 - x1
    delta_y = y2 - y1
    print(x1, y1, x2, y2, delta_x, delta_y)

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

            if max_val < 0.95:
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

if __name__ == '__main__':
    try:
        main(sys.argv[1], sys.argv[2])

    except:
        print('Error: ', sys.exc_info()[0])
        print(sys.exc_info()[1])
        import traceback
        print(traceback.format_tb(sys.exc_info()[2]))

    # except Exception as e:
    #     print(e)
