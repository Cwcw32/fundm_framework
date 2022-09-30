import os

import pytesseract

from PIL import Image

import cv2



# 使用python中的OCR库来进行OCR操作
class OCR_1(object):
    def __init__(self):
        pass
    def get_text(self,image_path,image_type="thresh",output_type=0):
        """
        :param image_path:图像的路径
        :param image_type:cv2读取的方法
        :return: 检测到的图像中的文本，空即为没有
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we should apply thresholding to preprocess the
        # image
        if image_type == "thresh":
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # make a check to see if median blurring should be done to remove
        # noise
        elif image_type == "blur":
            gray = cv2.medianBlur(gray, 3)

        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)
        # load the image as a PIL/Pillow image, apply OCR, and then delete
        # the temporary file
        text = pytesseract.image_to_string(Image.open(filename))
        # os.remove(filename)
        print(text)
        return  text


# 使用paddle-ocr来进行OCR操作
# 见本目录下的test2.py
