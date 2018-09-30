import cv2
import os

def cartoonify(img_rgb):
    numBilateralFilters = 7  # number of bilateral filtering steps

    # -- STEP 1 --
    img_color = img_rgb
    # repeatedly apply small bilateral filter instead of applying
    # one large filter
    for _ in range(numBilateralFilters):
        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
    print(img_color.shape)

    # -- STEPS 2 and 3 --
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    print(img_gray.shape)
    img_blur = cv2.medianBlur(img_gray, 7)
    print(img_blur.shape)

    # -- STEP 4 --
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 9, 2)
    print(img_edge.shape)

    # -- STEP 5 --
    # convert back to color so that it can be bit-ANDed with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    print(img_edge.shape)
    return cv2.bitwise_and(img_color, img_edge)

folder = 'Opencountry'
for filename in os.listdir(folder):
    img_rgb = cv2.imread(os.path.join(folder,filename))
    # If input file is img.jpg, name the output as img_cartoon.jpg
    split_filename = filename.split('.')
    outputFilename = split_filename[0] + '_cartoon.' + split_filename[1]

    if img_rgb is not None:
        output = cartoonify(img_rgb)
        cv2.imwrite(os.path.join(folder,outputFilename), output)
