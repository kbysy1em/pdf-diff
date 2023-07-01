import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt

posi = []

def onclick2(event):
    global posi

    # matplotlibでは横方向がx、縦方向がyになる模様
    posi.append([int(event.ydata), int(event.xdata)])
    print(f'Clicked point: {posi[-1]}')

class ImagePresenter:
    def __init__(self, settings):
        self.settings = settings

    def show(self, img1, img2, similarity1, similarity2,):
        global posi
        color_img = cv2.merge((img2, img2, img2))
        start_x = int(self.settings['intr_area_x'] * self.settings['step_x'])
        end_x = int(img2.shape[0] - 2 * self.settings['intr_area_x'] * self.settings['step_x'])
        start_y = int(self.settings['intr_area_y'] * self.settings['step_y'])
        end_y = int(img2.shape[1] - 2 * self.settings['intr_area_y'] * self.settings['step_y'])

        color_img_sub = color_img[start_x:end_x, start_y:end_y]
        img1_sub = img1[start_x:end_x, start_y:end_y]
        img2_sub = img2[start_x:end_x, start_y:end_y]
        similarity1_sub = similarity1[start_x:end_x, start_y:end_y]
        similarity2_sub = similarity2[start_x:end_x, start_y:end_y]
        
        color_img_sub[(img2_sub < 128) & (similarity1_sub < self.settings['criterion'])] = (255, 0, 0)
        color_img_sub[(img2_sub >= 128) & (similarity1_sub < self.settings['criterion'])] = (255, 192, 203)
        color_img_sub[(similarity1_sub >= self.settings['criterion']) & (img1_sub < 128) & (similarity2_sub < self.settings['criterion'])] = (0, 255, 0)
        color_img_sub[(similarity1_sub >= self.settings['criterion']) & (img1_sub >= 128) & (similarity2_sub < self.settings['criterion'])] = (152, 251, 152)

        color_img[start_x:end_x, start_y:end_y] = color_img_sub
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
                ax1.imshow(img1, vmin=0, vmax=255, cmap='gray')
                
                ax2 = plt.subplot(1, 2, 2)
                ax2.axis('off')
                ax2.set_position(mpl.transforms.Bbox([[0.5, 0], [1, 1]]))
                ax2.imshow(color_img)

            else:
                fig = plt.figure(figsize=(8.27, 11.69))
                
                ax1 = fig.add_subplot(2, 1, 1)
                ax1.axis('off')
                ax1.set_position(mpl.transforms.Bbox([[0, 0.5], [1, 1]]))
                ax1.imshow(img1, vmin=0, vmax=255, cmap='gray')
                
                ax2 = plt.subplot(2, 1, 2)
                ax2.axis('off')
                ax2.set_position(mpl.transforms.Bbox([[0, 0], [1, 0.5]]))
                ax2.imshow(color_img)

            plt.savefig('result.pdf', dpi=600, bbox_inches='tight')
            fig.canvas.mpl_connect('button_press_event', onclick2)
            plt.show()

            if len(posi) < 2:
                break

            if input('終了してよろしいですか?(y/n)') != 'n':
                break

            x2, y2 = posi.pop()
            x1, y1 = posi.pop()

            print(x1, x2, y1, y2)
            color_img[x1:x2, y1:y2] = cv2.merge((img2[x1:x2, y1:y2], img2[x1:x2, y1:y2], img2[x1:x2, y1:y2]))
