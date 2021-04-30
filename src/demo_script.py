import os
import os.path as osp

if __name__ == '__main__':
    video_dir = osp.expanduser("~/qinyu/xicheng")
    videos = os.listdir(video_dir)
    output_dir = osp.expanduser("~/qinyu/demo/gt")
    if not osp.exists(output_dir):
        os.system(f"mkdir -p {output_dir}")

    for video in videos:
        input_video = osp.join(video_dir, video)
        out_dir = osp.join(output_dir, video.split(".")[0])
        print(f"Perform tracking for {input_video}")
        cmd_str = f"python demo.py mot --load_model ../models/fairmot_dla34.pth --conf_thres 0.4 --input-video {input_video} --output-root {out_dir} --use_gt_box --gt_box_file ~/qinyu/xicheng.json"
        os.system(cmd_str)

    print("Done.")
