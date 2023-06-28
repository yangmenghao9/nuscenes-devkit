from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.utils.splits import val, mini_val
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
import argparse
import random
import os

def nuscenes_display():
    
    nusc = NuScenes(version=version_, dataroot=dataroot_, verbose=True)

    eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
        'v1.0-test': 'test'
        }

    eval_version = 'detection_cvpr_2019'
    eval_config = config_factory(eval_version)

    pred_results, _ = load_prediction(result_path_, 500, DetectionBox)
    gt_results = load_gt(nusc, eval_set_map[version_], DetectionBox)
    gt_results = add_center_dist(nusc, gt_results)
    gt_results = filter_eval_boxes(nusc, gt_results, eval_config.class_range)

    if version_ == "v1.0-trainval":
        scenes = random.sample(val, sample_num_)
    else:
        scenes = random.sample(mini_val, sample_num_)
    print("display scenes: ", scenes)
    for scene in scenes:
        output_path = None
        if output_dir_:
            if not os.path.isdir(output_dir_):
                os.mkdir(output_dir_)
            output_path = os.path.join(output_dir_, "%s.avi"%scene)
        my_scene_token = nusc.field2token('scene', 'name', scene)[0]

        if not show_:
            print("Processing %s, but not displaying"%scene)
        # display multi channel prediction
        nusc.render_scene_prediction(my_scene_token, 
                                    out_path=output_path, 
                                    display_ground_truth=compare_gt_, 
                                    pred_results=pred_results,
                                    gt_results=gt_results,
                                    socre=score_,
                                    show=show_)


if __name__ == "__main__":
    
    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Folder to store visualizations video.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--score', type=float, default=0.15,
                        help='Filter the score of the target')
    parser.add_argument('--sample_num', type=int, default=2,
                        help='The number you want to output')
    parser.add_argument('--compare_gt', action='store_true', help='Compare with ground truth')
    parser.add_argument('--show', action='store_true', help='Display results')
    args = parser.parse_args()

    result_path_ = args.result_path
    output_dir_ = args.output_dir
    dataroot_ = args.dataroot
    version_ = args.version
    score_ = args.score
    sample_num_ = args.sample_num
    compare_gt_ = args.compare_gt
    show_ = args.show

    nuscenes_display()