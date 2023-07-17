import os
from pypdf import PdfWriter

class Merger():
    def __init__(self, settings):
        self.merger = PdfWriter()
        self.settings = settings

    def execute(self, page_num=0):
        '''
        PDFファイルの結合を行う
        '''

        filename = f'{self.settings["output_filename"]}{page_num:02}.pdf'
        print(filename)
        # if filenameが存在しないならばファイルを保存してクローズ
        if not os.path.isfile(filename):
            self.merger.write(f'{self.settings["output_filename"]}.pdf')
            self.merger.close()
            return
        
        # それ以外の場合にはfilenameのファイルを結合
        self.merger.append(filename)

        # filenameとpage_numを引数にしてexecuteを再帰実行
        self.execute(page_num + 1)