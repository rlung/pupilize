#!/usr/bin/env python

import numpy as np
import h5py as h5
import pandas as pd
import cv2
import os
from time import strftime
from datetime import datetime
import glob
import multiprocessing as multi
from functools import partial
from argparse import ArgumentParser, RawTextHelpFormatter
import custom

import pdb


def pupilize(im, threshold=127, invert=False,
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


def pupil_trials(df, level=-1, bin_size=200, pre=-16000, post=21000):

    pre_bins = pre / bin_size
    post_bins = post / bin_size

    # Find unique animals from certain pattern in experiment names
    # subjects = np.unique([x.split('_')[0] for x in df.columns.levels[0]])

    exps = df.xs('trials', axis=1, level=-1).columns
    exps_dict = {}

    # Iterate over experiments (different planes are separate experiments)
    for exp in exps:
        epochs_dict = {}
        
        # Iterate over epochs
        for epoch in df[exp].index.levels[0]:
            df_epoch = df.loc[epoch, exp]
            trial_starts = df_epoch[df_epoch['trials'] == 1].index
            trial_window = zip(trial_starts + pre, trial_loc + post - bin_size)
            trials_dict = {}      # Initialize list of dataframes for each trial

            # Iterate over trials
            for n, x in enumerate(trial_window):
                # Get subset of dataframe for current trial window
                trial = df_epoch.loc[slice(*x), :]

                # Define new index (based on time relative to trial)
                # Need to define index relative to current index in case of missing data.
                index_vals = np.arange(pre, post, bin_size)
                trial_index = pd.DataFrame({
                    'trial_time': index_vals,
                    'time': np.linspace(*(x + (len(index_vals), )), dtype=int)
                })
                trial_index = trial_index.set_index('time')

                # Add and set new index
                trial = pd.concat([trial, trial_index], axis=1)
                trial = trial.reset_index(level='time')
                trial = trial.set_index('trial_time')
                trials_dict[n] = trial

            trials_df = pd.concat(trials_dict, names=['trial'], axis=1)
            epochs_dict[epoch] = trials_df
            
        epochs_df = pd.concat(epochs_dict, names=['epoch'], axis=0)
        exps_dict[exp] = epochs_df
        
    exps_df = pd.concat(exps_dict, names=exps.names, axis=1)
    exps_df = exps_df.sort_index(axis=1)  # Allows for indexing (killed myself over this)

    return exps_df

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


def manage_data(data, n_cores=1, threshold=127, sample_period=200, key='cam/frames', time_limit=300000):
    global df
    epoch_dict = {'1': 'base', '2': 'ctrl', '3': 'stim'}

    print("Analyzing {}...".format(os.path.basename(data)))

    _, animal_id, exp_day, plane, epoch = os.path.splitext(
        os.path.basename(data))[0].split('_')

    with h5.File(data, 'r') as hf:
        pupil_frames = hf[key]
        pupil_timestamps = hf[os.path.dirname(key) + '/timestamps']

        # Calculate pupil diameter
        pfunc = partial(pupilize, threshold=127, kernel=np.ones((7, 7)))
        if n_cores > 1:
            p = multi.Pool(processes=n_cores)
            boxes, _ = zip(*p.map(pfunc, pupil_frames))
        else:
            boxes, _ = zip(*map(pfunc, pupil_frames))
        pupil_diam = [w for _, _, w, _ in boxes]

        # Resample data
        # ts_max = min(time_limit, int(pupil_timestamps[-1]))
        # ts_new = np.arange(0, ts_max, sample_period)
        ts_new = np.array(df.index.levels[1])
        pupil_resampled = custom.resample(pupil_diam, pupil_timestamps, ts_new, method=np.mean)

        behav = hf['behavior']
        trials = custom.resample(
            np.ones(behav['trials'].shape), behav['trials'], ts_new, method=np.any)
        rail_home = custom.resample(
            np.ones(behav['rail_home'].shape), behav['rail_home'], ts_new, method=np.any)
        rail_leave = custom.resample(
            np.ones(behav['rail_leave'].shape), behav['rail_leave'], ts_new, method=np.any)
        track = custom.resample(
            behav['track'][1], behav['track'][0], ts_new, method=np.sum)

    # Save data
    col_name = (animal_id, exp_day, plane)
    df.set_value(epoch_dict[epoch], col_name + ('pupil', ), pupil_resampled)
    df.set_value(epoch_dict[epoch], col_name + ('trials', ), trials)
    df.set_value(epoch_dict[epoch], col_name + ('rail_home', ), rail_home)
    df.set_value(epoch_dict[epoch], col_name + ('rail_leave', ), rail_leave)
    df.set_value(epoch_dict[epoch], col_name + ('track', ), track)

    # df.set_value(ix0, col_name, pupil_resampled)
    # if df is None:
    #     series = {col_name: pupil_resampled}
    #     df = pd.DataFrame(
    #         series,
    #         index=pd.MultiIndex.from_tuples(
    #             zip([epoch] * len(ts_new), ts_new)
    #         )
    #     )
    # else:
    #     df[col_name, epoch] = pupil_resampled

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
        "data",
        help="HDF5 file (or directory containing files) with pupil data"
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
        "-n", "--number-of-cores", default=None,
        help="Number of cores to use for parallel processing"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output HDF5 file for data"
    )
    opts = parser.parse_args()
    thresh = int(opts.threshold)

    # Create DataFrame
    time_limit = 300000
    bin_size = 200
    ts = np.arange(0, time_limit, bin_size)
    
    nbins = len(ts)
    global df
    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            zip(['base'] * nbins, ts) + zip(['ctrl'] * nbins, ts) + zip(['stim'] * nbins, ts),
            names=['epoch', 'time']
        ),
        columns=pd.MultiIndex(
            levels=[[], [], [], []],
            labels=[[], [], [], []],
            names=['subject', 'experiment', 'plane', 'feature']
        )
    )

    # Process file(s)
    if os.path.isdir(opts.data):
        files = glob.glob(os.path.join(opts.data, '*.h5'))

        # pupils = np.column_stack([manage_data(f, mpool=p, df=df) for f in files])

        pfn = partial(manage_data, n_cores=int(opts.number_of_cores), threshold=thresh, key=opts.key)
        map(pfn, files)
    elif os.path.isfile(opts.data):
        manage_data(data, n_cores=int(opts.number_of_cores), threshold=thresh, key=opts.key)
    else:
        raise IOError("Invalid input for data")

    # Save DataFrame
    # np.savetxt("pupils.txt", pupils)
    if opts.output:
        outfile = opts.output
    else:
        outfile = 'pupils.h5'

    df = df.sort_index(axis=1)
    with pd.HDFStore(outfile) as hf:
    	hf['pupils'] = df
        hf.get_storer('pupils').attrs['threshold'] = thresh
        hf.get_storer('pupils').attrs['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("All done")


if __name__ == "__main__":
    main()
