import cv2

def detection(img_1, img_2):
    img_1 = cv2.imread('T35UQS_20231121T091301_B04_10m.tiff', cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread('T36UUB_20231121T091301_B04_10m.tiff', cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(img_1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img_2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img_1, keypoints1, img_2, keypoints2, matches[:10], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img3