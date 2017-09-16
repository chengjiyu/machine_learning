import numpy as np
from sklearn.cluster import KMeans
import PIL.Image as image
def loadData(filePath):
	data = []
	img = image.open(filePath)
	m, n = img.size
	for i in range(m):
		for j in range(n):
			x,y,z,a = img.getpixel((i, j))
			data.append([x/256.0,y/256.0,z/256.0])
	return np.mat(data),m,n
imgData, row, col = loadData("E:\\chengjiyu\\MLpython\\spurs.png")

km = KMeans(n_clusters=3)
label = km.fit_predict(imgData)
label = label.reshape([row, col])
pic_new = image.new("L", (row, col))
print(label)
for i in range(row):
	for j in range(col):
		pic_new.putpixel((i,j),256/(label[i][j]+1))

pic_new.save("result.jpg", "JPEG")
5 2
A B C
C F *
B D E
D G *
E H I