import matplotlib.pyplot as plt
from build_algorithm import detection

picture_1 = 'T35UQS_20231121T091301_B04_10m.tiff'
picture_2 = 'T36UUB_20231121T091301_B04_10m.tiff'

res = detection(picture_1, picture_2)
plt.figure(figsize=(15, 15))
plt.imshow(res)
plt.show()