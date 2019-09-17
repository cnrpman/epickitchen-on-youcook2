import os
import os.path
import sys
import shutil
import argparse
import math
from tqdm import tqdm

def opts():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', default='./',
                        help='path of decoded images')
    parser.add_argument('--path_dst', default='../imgs_cliped',
                        help='path of decoded images')
    parser.add_argument('--path_fr', default='../framerates.txt',
                        help='path of decoded images')
    parser.add_argument('--path_manifest', default='../clip_manifest.txt',
                        help='path of cliped manifest')  
    parser.add_argument('--duration', default=10, type=int,
                        help='duration for each clip, by second')
    parser.add_argument('--overlap', action='store_true',
                        help='overlapping between current and last clip')
    parser.add_argument('--img_tmpl', default='img_{:05d}.jpg',
                        help="image name template")
    return parser.parse_args()

def main():
    args = opts()
    record = list()
    
    framerates = [line.strip().split(' ') for line in open(args.path_fr, 'r')]
    vid2framerate = {os.path.splitext(line[0])[0] : float(line[1]) for line in framerates}

    if not os.path.exists(args.path_dst):
        os.mkdir(args.path_dst)

    img_list = os.listdir(args.path)
    for img in tqdm(img_list):
        framerate = vid2framerate[os.path.splitext(img)[0]]
        tot_frame = len(os.listdir(os.path.join(args.path, img)))
        
        st_fr = 1
        lst_fr = -1
        if args.overlap:
            duration = args.duration / 2
            cnt = 2
        else:
            duration = args.duration
            cnt = 1
        while(True):
            nxt_sec = cnt * duration
            nxt_fr = min(math.floor(nxt_sec * framerate), tot_frame+1)
            if st_fr == nxt_fr:
                break
            if args.overlap and lst_fr != -1:
                cut(args, img, cnt-1, lst_fr, nxt_fr, framerate, record)
            else:
                cut(args, img, cnt, st_fr, nxt_fr, framerate, record)
            cnt += 1

            lst_fr = st_fr
            st_fr = nxt_fr

    with open(args.path_manifest, 'w') as f:
        f.write('\n'.join(record))

def cut(args, img, cnt, start, end, framerate, record):
    skip_flg = False
    src_img = os.path.join(args.path, img)
    dst_img = os.path.join(args.path_dst, img + ("_%04d" % cnt))
    if not os.path.exists(dst_img):
        os.mkdir(dst_img)
    else:
        if len(os.listdir(dst_img)) == end - start:
            skip_flg = True
    
    if not skip_flg:
        for idx, fr in enumerate(range(start, end)):
            src_path = os.path.join(src_img, args.img_tmpl.format(fr))
            dst_path = os.path.join(dst_img, args.img_tmpl.format(idx+1))
            shutil.move(src_path, dst_path)
    record.append(' '.join((img + ("_%04d" % cnt), img, ("%.4f" % (start / framerate)), ("%.4f" % (end / framerate)))))

if __name__ == "__main__":
    main()
    
