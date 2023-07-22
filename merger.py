import os
from pypdf import PdfWriter

class Merger():
    def __init__(self, settings):
        self.merger = PdfWriter()
        self.settings = settings

    def execute(self, page_num=0):
        """PDFファイルの結合を行う
        """

        filename = f'{self.settings["output_filename"]}{page_num:02}.pdf'
        # if filenameが存在しないならばファイルを保存してクローズ
        if not os.path.isfile(filename) or page_num == self.settings['total_images1']:
            print('保存しています...', end=' ')
            self.merger.write(f'{self.settings["output_filename"]}.pdf')
            self.merger.close()
            print('完了')
            return

        # それ以外の場合にはfilenameのファイルを結合
        print(f'{filename}を結合します')
        self.merger.append(filename)

        # filenameとpage_numを引数にしてexecuteを再帰実行
        self.execute(page_num + 1)