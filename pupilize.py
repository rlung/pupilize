#!/usr/bin/env python

import numpy as np
import cv2


def find_pupil(im, threshold=127, invert=False, morph_kernel=np.ones((7, 7)), morph_iter=4):
    '''Finds and measures pupil
    Threshold image to identify pupil. High contrast is best to identify. 
    Morphological processing cleans up the pupil and a bounding box is defined 
    for the largest contour in the image.
    '''

    # Default values
    # max_area=25000
    
    # CONTOUR METHODS
    im = cv2.medianBlur(im, 5)
    frame = np.zeros(im.shape, dtype=np.uint8)
    frame[im >= threshold] = 255
    if invert:
        frame = 255 - frame

    # Isolate largest contour
    contours, _ = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (np.nan, ) * 4, None
    areas = [cv2.contourArea(contour) for contour in contours]
    contour_ix = np.argmax(areas)
    frame = np.zeros(frame.shape, dtype=np.uint8)
    cv2.drawContours(frame, contours, contour_ix, 255, -1)
    
    # Morphological processing
    frame = cv2.dilate(frame, morph_kernel, iterations=morph_iter)
    frame = cv2.erode(frame, morph_kernel, iterations=morph_iter * 2)
    frame = cv2.dilate(frame, morph_kernel, iterations=morph_iter)

    # Get contour again
    contours, _ = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (np.nan, ) * 4, None
    areas = [cv2.contourArea(contour) for contour in contours]

    # Calculate size of largest contour
    largest_contour = contours[np.argmax(areas)]
    # circle = cv2.minEnclosingCircle(largest_contour)
    # ellipse = cv2.fitEllipse(largest_contour)
    box = cv2.boundingRect(largest_contour)
    
    return box, largest_contour

    # HOUGH METHOD
    # im = cv2.medianBlur(im, 15)
    # circles = cv2.HoughCircles(im, cv2.cv.CV_HOUGH_GRADIENT,
    #     dp=1, minDist=100, param1=30, param2=30, minRadius=30, maxRadius=125)
    # if not np.any(circles):
    #     hough = None
    # else:
    #     radii = [circle[2] for circle in circles[0]]
    #     hough = circles[0][np.argmax(radii)]

    # return hough


def box2roi(box):
    x, y, w, h = box
    tl = [x, y]
    tr = [x + w, y]
    br = [x + w, y + h]
    bl = [x, y + h]
    
    return np.array([tl, tr, br, bl, tl])


def circle2roi(circles):
    if circles is None:
        return np.array([0, 0, 0, 0])

    x, y, r = circles
    tl = [x - r, y - r]
    tr = [x + r, y - r]
    br = [x + r, y + r]
    bl = [x - r, y + r]

    return np.array([tl, tr, br, bl, tl])
