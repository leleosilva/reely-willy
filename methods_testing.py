import numpy as np
import cv2

base_img = cv2.imread('templates/stardew.png')
template = cv2.imread('templates/catch.png')
h, w = template.shape[0], template.shape[1]

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

# cv2.TM_CCOEFF_NORMED and cv2.TM_SQDIFF_NORMED seem to work better (cleaner results) with the chosen template
for method in methods:
    img = base_img.copy()

    result = cv2.matchTemplate(img, template, method)
    _, _, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)    
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 3)
    cv2.imshow('Detected point', img)
    cv2.imshow('Matching result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Might not work on terminal out of interactive mode without opencv-python-headless installed.
    # Note: cv2.TM_SQDIFF result not displayed properly when using pyplot as below.
    """
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 8))
    axs[0] = plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Matching result'), plt.xticks([]), plt.yticks([])
    axs[1] = plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Detected point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(method)
    plt.show()
    """