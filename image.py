import PIL.Image
from PIL import Image

image = Image.open('C:\\Users\\Lenovo\\Pictures\\新建文件夹\\22.jpg')
#open()函数用于从文件加载图像，如果成功，会返回一个Image类
# image.show()

# image.save('1.jpg')
print(image.mode, image.size, image.format)
#format表示了图像来源，若如果图像不是从文件读取它的值就是None
#size为包含宽度和高度的二元组
#mode定义图像段波的数量和名称

image=PIL.Image.alpha_composite(image,image)
#返回一个新的image对象，且两个图像必须要哟rgba模式


