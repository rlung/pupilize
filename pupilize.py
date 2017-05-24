#!/usr/bin/env python

import numpy as np
import h5py as h5
import pandas as pd
import cv2
import os
import glob
import multiprocessing as multi
from functools import partial
from argparse import ArgumentParser, RawTextHelpFormatter
import custom

import pdb


def pupilize(im, threshold=127, invert=True,
             kernel=np.ones((7, 7)), morph_iter=4):
    '''
    Measures pupil.
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
    frame = cv2.dilate(frame, kernel, iterations=morph_iter)
    frame = cv2.erode(frame, kernel, iterations=morph_iter * 2)
    frame = cv2.dilate(frame, kernel, iterations=morph_iter)

    # Get contour again
    contours, _ = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (np.nan, ) * 4, None
    areas = [cv2.contourArea(contour) for contour in contours]

    # Calculate circle from largest contour
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


def manage_data(data, mpool=None, sample_period=200, key='cam/frames'):
    global df

    print("Analyzing {}...".format(os.path.basename(data)))

    animal_id, exp_day, stim, date, epoch = os.path.splitext(
        os.path.basename(data))[0].split('_')

    hf = h5.File(data, 'r')
    pupil_frames = hf[key]
    pupil_timestamps = hf[os.path.basename(key) + '/timestamps']

    # Calculate pupil diameter
    pfunc = partial(pupilize, threshold=127, kernel=np.ones((7, 7)))
    boxes, _ = zip(*mpool.map(pfunc, pupil_frames))
    pupil_diam = [w for _, _, w, _ in boxes]

    # Resample data
    ts_new = np.arange(0, int(pupil_timestamps[-1]), sample_period)
    pupil_resampled = custom.resample(pupil_diam, pupil_timestamps, ts_new)

    # Save data
    col_name = '{}_{}_{}_{}'.format(animal_id, exp_day, stim, epoch)
    df = df.assign(**{col_name: pupil_resampled})

    # save_file = '{}_{}_{}.txt'.format(animal_id, exp_day, epoch)
    # np.savetxt(save_file, pupil_diam)
    # print("Saved data to {}".format(save_file))

    print("Finished")


def main():
    parser = ArgumentParser(
        description="Calculate pupil size",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "data", help="HDF5 file (or directory containing files) with pupil data"
    )
    parser.add_argument(
        "-k", "--key", default='cam/frames',
        help="HDF5 key for pupil frames"
    )
    parser.add_argument(
        "-t", "--threshold", default=95,
        help="Threshold for creating binary pupil image"
    )
    parser.add_argument(
        "-n", "--number-of-cores", default=1,
        help="Number of cores to use for parallel processing"
    )
    # parser.add_argument(
    #     "-a", "--append", default='false', action='store_true',
    #     help="Appends data to HDF5 file"
    # )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output HDF5 file for data"
    )
    opts = parser.parse_args()

    # Setup multiprocessing pool
    p = multi.Pool(processes=int(opts.number_of_cores))

    # Process file(s)
    global df
    df = pd.DataFrame()
    if os.path.isdir(opts.data):
        files = glob.glob(os.path.join(opts.data, '*.h5'))

        # pupils = np.column_stack([manage_data(f, mpool=p, df=df) for f in files])

        pfn = partial(manage_data, mpool=p, key=opts.key)
        map(pfn, files)
    elif os.path.isfile(opts.data):
        manage_data(data, mpool=p, key=opts.key)
    else:
        raise IOError("Invalid input for data")

    # Save data
    # np.savetxt("pupils.txt", pupils)
    df = df.sort_index(axis=1)
    df.assign(time=np.arange(len(df)) * 200).set_index('time')
    df_h5 = pd.HDFStore('pupils.h5')
    df_h5['pupils'] = df
    df_h5.close()

    print("All done")


if __name__ == "__main__":
    main()
