from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq


logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    if opt.box_json:
        assert dataloader.vn == len(opt.boxes), f"Error: frame_count ({dataloader.vn}) != len(opt.boxes) ({len(opt.boxes)})"
        dataloader.boxes = opt.boxes

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        if opt.box_json:
            video_name = (opt.input_video.split("/")[-1]).split(".")[0]
            det_name = (opt.box_json.split("/")[-1]).split(".")[0]
            output_video_path = osp.join(result_root, f'{video_name}_{det_name}.mp4')
        else:
            output_video_path = osp.join(result_root, 'MOT16-03-results.mp4')
        fps = frame_rate if opt.sample_rate == 1 else frame_rate // opt.sample_rate
        cmd_str = f"ffmpeg -threads 2 -y -f image2 -r {fps} -i {osp.join(result_root, 'frame')}/%05d.jpg -b:v 5000k -c:v mpeg4 {output_video_path}"
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
