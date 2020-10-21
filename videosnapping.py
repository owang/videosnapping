"""
This is a personal, partial reimplemtation of:

VideoSnapping: interactive synchronization of multiple videos
Oliver Wang, Christopher Schroers, Henning Zimmer, Markus H. Gross, Alexander Sorkine-Hornung

Please see the README.md before using.

"""

import glob
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mpe
import numpy as np
from tqdm import tqdm

import configargparse

# for moviepy video visualization
ffmpeg_params = [
    '-crf', '5', '-pix_fmt', 'yuv420p', '-vf', 'pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2'
]


def parse_arguments():
    parser = configargparse.ArgumentParser(description='videosnapping')
    parser.add('-c', '--config', is_config_file=True, help='config file path')
    parser.add("--vid1", default=None, type=Path, help="input video 1")
    parser.add("--vid2", default=None, type=Path, help="input video 2")
    parser.add("--output-path", default=Path('./output'), type=Path, help="path to dump outputs")
    parser.add("--fps",
               default=10,
               type=int,
               help="Number of fps to run CM computation at. Higher = slower but more accurate.")
    parser.add(
        "--min-steps",
        default=40,
        type=int,
        help=
        "Mininum number of frames that has to be in correspondences between vids (only needed for --partial-path)."
        "Higher = more robust to corners, but may miss paths if the two videos have a small overlap"
    )
    parser.add(
        "--cm-sigma",
        default=400,
        type=float,
        help=
        "Sigma to convert the sift histogram to cost matrix. Higher = more tolerant of differences in sift histogram counts"
    )
    parser.add(
        "--cm-scale",
        default=1,
        type=float,
        help=
        "Weight for balancing the cost of a bad match vs the length of a path step in dynamic programing (higher = care more about the frame similarity)"
    )
    parser.add(
        "--cm-gamma",
        default=2,
        type=float,
        help=
        "Penalty function for high vs low cost (higher = more difference between hitting high and low quality matches)"
    )
    parser.add("--overwrite", action='store_true', help="Overwrite cost matrix path cache")
    parser.add("--visualize", action='store_true', help="Create a window showing the path")
    parser.add("--partial-alignment",
               action='store_true',
               help="Compute shortest average path, otherwise compute shortest path.")
    parser.add("--run-at-low-fps",
               action='store_true',
               help="Do not upscale the cost matrix, and run full code on low fps results")
    args = parser.parse_args()

    return args


def compute_shortest_path(cost_matrix, partial_alignment=False, min_steps=100):
    """ Finds either the shortest path from the start of the two frames
    or the the shortest average path through any number of frames (partial alignment)
    of a cost matrx using dyanmic programing.
    Inputs:
        min_steps: the mininum number of frames required to be in alignment
    """
    n1 = cost_matrix.shape[0]
    n2 = cost_matrix.shape[1]

    # create intermediate storage
    path_cost_matrix = np.empty((n1, n2))
    path_cost_matrix[:] = np.inf
    prev_index = np.zeros((n1, n2, 2), np.int)
    num_steps = np.zeros((n1, n2), np.int)

    if partial_alignment:
        # start at any frame from either video
        for f1 in range(n1):
            path_cost_matrix[f1, 0] = cost_matrix[f1, 0]
            prev_index[f1, 0, :] = -1
        for f2 in range(n2):
            path_cost_matrix[0, f2] = cost_matrix[0, f2]
            prev_index[0, f2, :] = -1
    else:
        # start at the first frame of both videos
        path_cost_matrix[0, 0] = cost_matrix[0, 0]
        prev_index[0, 0, :] = -1

    # build the graph edges, connecting forward in time
    for f1 in range(n1):
        for f2 in range(n2):
            # valid movements for frame mapping (advance either, or both videos one frame)
            neighbors = []
            if f1 > 0:
                neighbors.append((f1 - 1, f2))
            if f2 > 0:
                neighbors.append((f1, f2 - 1))
            if f1 > 0 and f2 > 0:
                neighbors.append((f1 - 1, f2 - 1))

            # check costs of the neighbor connections to the previous frame
            for p1, p2 in neighbors:
                edge_cost = 1
                prev_cost = path_cost_matrix[p1, p2]
                my_cost = cost_matrix[f1, f2]
                prev_steps = num_steps[p1, p2]

                # my cost is previous best cost plus edge cost plus my cost
                cost = prev_cost + edge_cost + my_cost
                if cost < path_cost_matrix[f1, f2]:
                    path_cost_matrix[f1, f2] = cost
                    prev_index[f1, f2, 0] = p1
                    prev_index[f1, f2, 1] = p2
                    num_steps[f1, f2] = prev_steps + 1

    if partial_alignment:
        # find the start of the path
        shortest_path = None
        path_cost = np.inf
        for f1 in range(n1):
            average_path_cost = path_cost_matrix[f1, n2 - 1] / num_steps[f1, n2 - 1]
            if average_path_cost < path_cost and num_steps[f1, n2 - 1] > min_steps:
                path_cost = average_path_cost
                shortest_path = (f1, n2 - 1)
        for f2 in range(n2):
            average_path_cost = path_cost_matrix[n1 - 1, f2] / num_steps[n1 - 1, f2]
            if average_path_cost < path_cost and num_steps[n1 - 1, f2] > min_steps:
                path_cost = average_path_cost
                shortest_path = (n1 - 1, f2)

        if shortest_path is None:
            print('No path found: try lowering --min-steps')
            sys.exit(1)

        shortest_path = np.asarray([shortest_path])
    else:
        # travel backwards through best path matrix to find the final best path
        # start at the end of both videos
        shortest_path = np.asarray([[n1 - 1, n2 - 1]])
        path_cost = path_cost_matrix[n1 - 1, n2 - 1]

    # travel backwards through best path matrix to find the final best path
    done = False
    while not done:
        prev = prev_index[shortest_path[-1, 0], shortest_path[-1, 1], :]
        if prev[0] == -1:
            done = True
        else:
            shortest_path = np.concatenate((shortest_path, prev[None, :]))

    # reverse the list
    shortest_path = shortest_path[::-1, :]

    return shortest_path, path_cost


def get_pair_string(vid1, vid2, args):
    """ this will serve as a unique identifier for a cost matrix """
    prefix = f'{vid1.name}_{vid2.name}_{args.fps}fps'
    return prefix


def get_cost_matrix(vid1, vid2, args):
    """ compute the cost matrices between two videos as a SIFT histogram
    (this is slow, should be parallelized) and dump them to file, so they
    can be read for subsequent steps (this is fast) """
    clip1 = mpe.VideoFileClip(str(vid1))
    clip2 = mpe.VideoFileClip(str(vid2))

    cache_fn = args.output_path / f'{get_pair_string(vid1,vid2,args)}.pkl'
    if not os.path.isfile(cache_fn) or args.overwrite:
        print(f'\tcomputing: {cache_fn}')
        num_frames1 = sum(1 for x in clip1.iter_frames(fps=args.fps))
        num_frames2 = sum(1 for x in clip2.iter_frames(fps=args.fps))
        histogram = np.zeros((num_frames1, num_frames2), dtype=np.float32)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=0,
                                           nOctaveLayers=3,
                                           contrastThreshold=0.001,
                                           edgeThreshold=20,
                                           sigma=1.6)

        # temporary caches to avoid recomputation
        clip1_sift = {}
        clip2_sift = {}

        for i1, frame1 in tqdm(enumerate(clip1.iter_frames(fps=args.fps)),
                               total=num_frames1,
                               desc=f'matching SIFT features: {get_pair_string(vid1, vid2, args)}',
                               dynamic_ncols=True):

            hash1 = f'{vid1}_{i1}'
            if hash1 in clip1_sift:
                kp1, des1 = clip1_sift[hash1]
            else:
                kp1, des1 = sift.detectAndCompute(frame1, None)
                clip1_sift[hash1] = (kp1, des1)

            bf = cv2.BFMatcher()

            # match frame1 to all frames in second video
            for i2, frame2 in enumerate(clip2.iter_frames(fps=args.fps)):
                hash2 = f'{vid2}_{i2}'
                if hash2 in clip2_sift:
                    kp2, des2 = clip2_sift[hash2]
                else:
                    kp2, des2 = sift.detectAndCompute(frame2, None)
                    clip2_sift[hash2] = (kp2, des2)

                matches = bf.knnMatch(des1, des2, k=2)

                # apply ratio test to remove bad matches
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])

                # add to histogram
                histogram[i1, i2] = len(good)

        pickle.dump(histogram, open(cache_fn, 'wb'))
    else:
        # histogram is computed, just load it
        print(f'\tloaded cache: {cache_fn}')
        histogram = pickle.load(open(cache_fn, 'rb'))
        num_frames1 = histogram.shape[0]
        num_frames2 = histogram.shape[1]

    # convert histogram to cost matrix
    cost_matrix = np.exp(-.5 * (histogram**2 / args.cm_sigma**2))

    # scale the cost matrix according to input parameters
    cost_matrix = (cost_matrix**args.cm_gamma) * args.cm_scale

    return cost_matrix


def log_path(cost_matrix, path, pair_string, args):
    """ display or write the path to file """
    path_filename = args.output_path / f'{pair_string}_path.png'
    plt.figure()
    plt.imshow(cost_matrix, cmap='hot', extent=[0, cost_matrix.shape[1], cost_matrix.shape[0], 0])
    for i in range(path.shape[0]):
        plt.plot(path[i, 1], path[i, 0], 'b.')
    plt.savefig(path_filename)
    if args.visualize:
        plt.show()
    plt.close()


def render_videos(vid1, vid2, shortest_path, pair_string, args):
    """ note this just renders each step along the path using nearest neighbor frames. """

    movie_filename = args.output_path / f'{pair_string}.mp4'

    # dump temporary output frames here (clear any that exist)
    tmp_folder = Path('./tmp_videosnapping_rendered_frames')
    if tmp_folder.is_dir():
        shutil.rmtree(tmp_folder)
    while tmp_folder.is_dir():
        pass
    tmp_folder.mkdir(exist_ok=False)

    clip1 = mpe.VideoFileClip(str(vid1))
    clip2 = mpe.VideoFileClip(str(vid2))
    num_frames1 = sum(1 for x in clip1.iter_frames())
    num_frames2 = sum(1 for x in clip2.iter_frames())

    fps_scale = shortest_path.shape[0] / num_frames1

    if args.run_at_low_fps:
        fps1 = args.fps
        fps2 = args.fps
    else:
        fps1 = clip1.fps
        fps2 = clip2.fps

    for f in range(shortest_path.shape[0]):
        f1 = shortest_path[f, 0]
        f2 = shortest_path[f, 1]
        i1 = clip1.get_frame((f1 / fps1))
        i2 = clip2.get_frame((f2 / fps2))
        cvim = np.concatenate((i1, i2), axis=1)
        cvim = cv2.cvtColor(cvim, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(tmp_folder / f'{f:05}.png'), cvim)

    # make images into a new movie
    mpe.ImageSequenceClip(sequence=str(tmp_folder),
                          fps=25).write_videofile(str(movie_filename),
                                                  ffmpeg_params=ffmpeg_params,
                                                  logger=None)


def videosnapping(args):

    print(f'VideoSnapping Demo')
    args.output_path.mkdir(exist_ok=True)

    # load videos
    pair_string = get_pair_string(args.vid1, args.vid2, args)
    print(f'\tprocessing: {pair_string}')

    cm = get_cost_matrix(args.vid1, args.vid2, args)
    print(f'\tcomputed cost matrix, dim: {cm.shape}')

    # upscale the path to be the input size so that the path is computed at full temporal resolution
    if not args.run_at_low_fps:
        with mpe.VideoFileClip(str(args.vid1)) as clip1, mpe.VideoFileClip(str(
                args.vid2)) as clip2:
            num_frames1 = sum(1 for x in clip1.iter_frames())
            num_frames2 = sum(1 for x in clip2.iter_frames())
            cm = cv2.resize(cm, (num_frames2, num_frames1), cv2.INTER_LINEAR)
        print(f'\tresized cost matrix, dim: {cm.shape}')

    shortest_path, path_cost = compute_shortest_path(cm,
                                                     partial_alignment=args.partial_alignment,
                                                     min_steps=args.min_steps)
    print(f'\tcomputed shortest path, cost: {path_cost}, length: {shortest_path.shape[0]}')

    log_path(cm, shortest_path, pair_string, args)
    render_videos(args.vid1, args.vid2, shortest_path, pair_string, args)
    print(f'\tfinished rendering {pair_string}')
    return path_cost


if __name__ == "__main__":
    args = parse_arguments()
    videosnapping(args)
