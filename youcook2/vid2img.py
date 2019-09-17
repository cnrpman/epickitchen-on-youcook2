# Author: Junyi Du (junyidu@usc.edu)
# Adopt from Code for "TSM: Temporal Shift Module for Efficient Video Understanding", by Ji Lin*, Chuang Gan, Song Han

from __future__ import print_function, division
import os
import sys
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

n_thread = 23 # Don't do evil


def vid2jpg(file_name, dir_path, dst_dir_path):
    if all(x not in file_name for x in [".mp4", ".mkv", ".webm"]):
        return
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_dir_path, name)

    video_file_path = os.path.join(dir_path, file_name)
    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'img_00001.jpg')):
                subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
                os.mkdir(dst_directory_path)
            else:
                print('*** convert has been done: {}'.format(dst_directory_path))
                return
        else:
            os.mkdir(dst_directory_path)
    except:
        print(f'skipping {dst_directory_path}')
        return
    cmd = 'ffmpeg -i \"{}\" -threads 1 -vf scale=-1:331 -q:v 0 \"{}/img_%05d.jpg\"'.format(video_file_path, dst_directory_path)
    # print(cmd)
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def class_process(dir_path, dst_dir_path, class_name=None):
    """
    class_name: if specific, use specific dir for each class for src / dst; else, don't create such specific dir
    """
    print('*' * 20, class_name, '*'*20)
    if class_name is not None:
        dir_path = os.path.join(dir_path, class_name)
        dst_dir_path = os.path.join(dst_dir_path, class_name)

    if not os.path.exists(dst_dir_path):
        os.mkdir(dst_dir_path)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"There's not a dir {dir_path}")
    
    vid_list = os.listdir(dir_path)
    vid_list.sort()
    p = Pool(n_thread)
    from functools import partial
    worker = partial(vid2jpg, dir_path=dir_path, dst_dir_path=dst_dir_path)
    for _ in tqdm(p.imap_unordered(worker, vid_list), total=len(vid_list)):
        pass
    # p.map(worker, vid_list)
    p.close()
    p.join()

    print('\n')


if __name__ == "__main__":
    path_raw_video = sys.argv[1]
    dst_dir_path = sys.argv[2]

    if not os.path.isdir(path_raw_video):
        raise FileNotFoundError(f"There's no such folder called {path_raw_video}")

    class_process(path_raw_video, dst_dir_path)