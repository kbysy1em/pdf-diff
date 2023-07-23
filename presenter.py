import cv2
import io
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

clicked_positions = []
clicked_ax = None

def onclick(event):
    global clicked_ax
    global clicked_positions

    clicked_ax = event.inaxes
    # matplotlibでは横方向がx、縦方向がyになる模様
    clicked_positions.append([int(event.ydata), int(event.xdata)])
    print(f'Clicked point: {clicked_positions[-1]}')



class ImagePresenter:
    def __init__(self, settings, page_num, img1, img2, pdfs, similarity1=None, similarity2=None):
        self.settings = settings
        self.img1 = img1
        self.img2 = img2
        self.page_num = page_num
        self.similarity1 = similarity1
        self.similarity2 = similarity2
        self.criterion = settings['criterion'] * (1 / settings['step_x']) * (1 / settings['step_y'])
        self.pdfs = pdfs

        self.color_img1 = cv2.merge((img1, img1, img1))
        self.color_img2 = cv2.merge((img2, img2, img2))

        self.start_x = int(self.settings['intr_area_x'])
        self.end_x = int(self.img2.shape[0] - 2 * self.settings['intr_area_x'])
        self.start_y = int(self.settings['intr_area_y'])
        self.end_y = int(self.img2.shape[1] - 2 * self.settings['intr_area_y'])

        self.color_img1_sub = self.color_img1[self.start_x:self.end_x, self.start_y:self.end_y]
        self.color_img2_sub = self.color_img2[self.start_x:self.end_x, self.start_y:self.end_y]
        self.img1_sub = self.img1[self.start_x:self.end_x, self.start_y:self.end_y]
        self.img2_sub = self.img2[self.start_x:self.end_x, self.start_y:self.end_y]

        if similarity1 is not None:
            self.similarity1_sub = self.similarity1[self.start_x:self.end_x, self.start_y:self.end_y]
        if similarity2 is not None:
            self.similarity2_sub = self.similarity2[self.start_x:self.end_x, self.start_y:self.end_y]

class ImagePresenterInverseLeft(ImagePresenter):
    def show(self):
        global clicked_ax
        global clicked_positions
        
        self.color_img2_sub[(self.img2_sub < 128) & (self.similarity2_sub < self.criterion)] = self.settings['red']
        self.color_img2_sub[(self.img2_sub >= 128) & (self.similarity2_sub < self.criterion)] = self.settings['pink']
        self.color_img1_sub[(self.img1_sub < 128) & (self.similarity1_sub < self.criterion)] = self.settings['deep_green']
        self.color_img1_sub[(self.img1_sub >= 128) & (self.similarity1_sub < self.criterion)] = (152, 251, 152)

        self.color_img1[self.start_x:self.end_x, self.start_y:self.end_y] = self.color_img1_sub
        self.color_img2[self.start_x:self.end_x, self.start_y:self.end_y] = self.color_img2_sub

        retry = True
        while retry:
            clicked_positions = []

            print('一部の着色箇所を白に戻すには、戻したい箇所(長方形2点)を選択した上で、画像を閉じ\n'
                + '次に現れるダイアログで n を入力してください')

            if self.img2.shape[0] > self.img2.shape[1]:
                fig = plt.figure(figsize=(11.69, 8.27))
                
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.axis('off')
                ax1.set_position(mpl.transforms.Bbox([[0, 0], [0.5, 1]]))
                ax1.imshow(self.color_img1)
                
                ax2 = plt.subplot(1, 2, 2)
                ax2.axis('off')
                ax2.set_position(mpl.transforms.Bbox([[0.5, 0], [1, 1]]))
                ax2.imshow(self.color_img2)

            else:
                fig = plt.figure(figsize=(8.27, 11.69))
                
                ax1 = fig.add_subplot(2, 1, 1)
                ax1.axis('off')
                ax1.set_position(mpl.transforms.Bbox([[0, 0.5], [1, 1]]))
                ax1.imshow(self.color_img1)
                
                ax2 = plt.subplot(2, 1, 2)
                ax2.axis('off')
                ax2.set_position(mpl.transforms.Bbox([[0, 0], [1, 0.5]]))
                ax2.imshow(self.color_img2)

            axs = {ax1:'ax1', ax2:'ax2'}
            
            try:
                plt.savefig(f'{self.settings["output_filename"]}{self.page_num:02}.pdf', dpi=600, bbox_inches='tight')
            except PermissionError:
                raise       
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()

            if len(clicked_positions) < 2:
                break

            if input('終了してよろしいですか?(y/n)') != 'n':
                break

            x2, y2 = clicked_positions.pop()
            x1, y1 = clicked_positions.pop()

            print(f'色を戻す範囲: x方向 {x1} ~ {x2}, y方向 {y1} ~ {y2}')

            if axs[clicked_ax] == 'ax1':
                self.color_img1[x1:x2, y1:y2] = cv2.merge((self.img1[x1:x2, y1:y2], self.img1[x1:x2, y1:y2], self.img1[x1:x2, y1:y2]))
            elif axs[clicked_ax] == 'ax2':
                self.color_img2[x1:x2, y1:y2] = cv2.merge((self.img2[x1:x2, y1:y2], self.img2[x1:x2, y1:y2], self.img2[x1:x2, y1:y2]))
            else:
                raise NotImplementedError()

class ImagePresenterInverseRight(ImagePresenter):
    def show(self):
        global clicked_positions
        
        self.color_img2_sub[(self.img2_sub < 128) & (self.similarity2_sub < self.criterion)] = self.settings['red']
        self.color_img2_sub[(self.img2_sub >= 128) & (self.similarity2_sub < self.criterion)] = self.settings['pink']
        self.color_img2_sub[(self.similarity2_sub >= self.criterion) & (self.img1_sub < 128) & (self.similarity1_sub < self.criterion)] = self.settings['deep_green']
        self.color_img2_sub[(self.similarity2_sub >= self.criterion) & (self.img1_sub >= 128) & (self.similarity1_sub < self.criterion)] = (152, 251, 152)

        self.color_img2[self.start_x:self.end_x, self.start_y:self.end_y] = self.color_img2_sub
        retry = True
        while retry:
            clicked_positions = []

            print('一部の着色箇所を白に戻すには、戻したい箇所(長方形2点)を選択した上で、画像を閉じ\n'
                + '次に現れるダイアログで n を入力してください')

            if self.img2.shape[0] > self.img2.shape[1]:
                fig = plt.figure(figsize=(11.69, 8.27))
                
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.axis('off')
                ax1.set_position(mpl.transforms.Bbox([[0, 0], [0.5, 1]]))
                ax1.imshow(self.img1, vmin=0, vmax=255, cmap='gray')
                
                ax2 = plt.subplot(1, 2, 2)
                ax2.axis('off')
                ax2.set_position(mpl.transforms.Bbox([[0.5, 0], [1, 1]]))
                ax2.imshow(self.color_img2)

            else:
                fig = plt.figure(figsize=(8.27, 11.69))
                
                ax1 = fig.add_subplot(2, 1, 1)
                ax1.axis('off')
                ax1.set_position(mpl.transforms.Bbox([[0, 0.5], [1, 1]]))
                ax1.imshow(self.img1, vmin=0, vmax=255, cmap='gray')
                
                ax2 = plt.subplot(2, 1, 2)
                ax2.axis('off')
                ax2.set_position(mpl.transforms.Bbox([[0, 0], [1, 0.5]]))
                ax2.imshow(self.color_img2)

            try:

                plt.savefig(f'{self.settings["output_filename"]}{self.page_num:02}.pdf', dpi=600, bbox_inches='tight')
            except PermissionError:
                raise       
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()

            if len(clicked_positions) < 2:
                break

            if input('終了してよろしいですか?(y/n)') != 'n':
                break

            x2, y2 = clicked_positions.pop()
            x1, y1 = clicked_positions.pop()

            print(f'色を戻す範囲: x方向 {x1} ~ {x2}, y方向 {y1} ~ {y2}')
            self.color_img2[x1:x2, y1:y2] = cv2.merge((self.img2[x1:x2, y1:y2], self.img2[x1:x2, y1:y2], self.img2[x1:x2, y1:y2]))

class ImagePresenterOverlap(ImagePresenter):
    """ImagePresenter class in Overlap mode
    """
    def show(self):
        print('比較元画像の位置を調整します')
        print(f'x方向移動量: {self.settings["delta_x"]} (下が正方向)')
        print(f'y方向移動量: {self.settings["delta_y"]} (右が正方向)')

        M = np.float32([[1, 0, self.settings['delta_x']], [0, 1, self.settings['delta_y']]])
        img1_offset = cv2.warpAffine(self.img1, M, (self.img1.shape[1], self.img1.shape[0]), borderValue=255)

        height, width = img1_offset.shape

        self.color_img1 = np.zeros((height, width, 4), dtype=np.uint8)
        self.color_img1[:, :, 2] = img1_offset  # 黒い部分を青に変換
        self.color_img1[:, :, 3] = np.where(img1_offset == 0, 255, 0)  # 黒い部分を不透明に

        self.color_img2 = np.zeros((height, width, 4), dtype=np.uint8)
        self.color_img2[:, :, 0] = self.img2  # 黒い部分を赤に変換
        self.color_img2[:, :, 3] = np.where(self.img2 == 0, 255, 0)  # 黒い部分を不透明に

        # ２枚の画像を重ね合わせる
        result = cv2.add(self.color_img1, self.color_img2)

        if self.img2.shape[0] > self.img2.shape[1]:  # if img2 is portrait
            plt.figure(figsize=(8.27, 11.69))
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
            plt.imshow(result)
        else:  # if img2 is landscape
            plt.figure(figsize=(11.69, 8.27))
            plt.axis('off')
            plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02) 
            plt.imshow(result)

        try:
            plt.savefig(f'{self.settings["output_filename"]}{self.page_num:02}.pdf', dpi=600, bbox_inches='tight')
        except PermissionError:
            raise

        plt.show()

class ImagePresenterRaw(ImagePresenter):
    """ImagePresenter class in Raw mode
    """
    def show(self):
        if self.img2.shape[0] > self.img2.shape[1]:
            fig = plt.figure(figsize=(11.69, 8.27))
            
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.axis('off')
            ax1.set_position(mpl.transforms.Bbox([[0, 0], [0.5, 1]]))
            ax1.imshow(self.img1, cmap='gray')
            
            ax2 = plt.subplot(1, 2, 2)
            ax2.axis('off')
            ax2.set_position(mpl.transforms.Bbox([[0.5, 0], [1, 1]]))
            ax2.imshow(self.img2, cmap='gray')

        else:
            fig = plt.figure(figsize=(8.27, 11.69))
            
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.axis('off')
            ax1.set_position(mpl.transforms.Bbox([[0, 0.5], [1, 1]]))
            ax1.imshow(self.img1, cmap='gray')
            
            ax2 = plt.subplot(2, 1, 2)
            ax2.axis('off')
            ax2.set_position(mpl.transforms.Bbox([[0, 0], [1, 0.5]]))
            ax2.imshow(self.img2, cmap='gray')

        try:
            pdf = io.BytesIO()
            plt.savefig(pdf, format='pdf', dpi=300, bbox_inches='tight')
            self.pdfs.append(pdf)
        except PermissionError:
            raise       
        plt.show()

        with open("test.pdf", "wb") as testtest:
            testtest.write(self.pdfs[0].getvalue())

