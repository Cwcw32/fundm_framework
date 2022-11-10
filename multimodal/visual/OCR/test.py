import pytesseract
try:
    from PIL import Image
except ImportError:
    import Image

# 列出支持的语言
print(pytesseract.get_languages(config=''))

print(pytesseract.image_to_string(Image.open('./images/test2.png'), lang='chi_sim+eng'))