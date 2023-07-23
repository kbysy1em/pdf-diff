import io
import os
from pypdf import PdfWriter

class Merger():
    def __init__(self, settings, pdfs):
        self.merger = PdfWriter()
        self.settings = settings
        self.pdfs = pdfs
        self.outputfile = io.BytesIO()

    def execute(self, page_num=0):
        """PDFファイルの結合を行う
        """
        print(page_num)
        with open("test2.pdf", "wb") as testtest:
            testtest.write(self.pdfs[0].getvalue())

        # if filenameが存在しないならばファイルを保存してクローズ
        if page_num == self.settings['total_images1']:
            print('保存しています...', end=' ')

#            with open(self.outputfile, "wb") as output:
            self.merger.write(self.outputfile)
            self.merger.close()
  
            with open('result.pdf', 'wb') as output:
                output.write(self.outputfile.getvalue())
                print('完了')
            return

        # それ以外の場合にはfilenameのファイルを結合
        self.merger.append(io.BytesIO(self.pdfs[page_num].getvalue()))

        # filenameとpage_numを引数にしてexecuteを再帰実行
        self.execute(page_num + 1)