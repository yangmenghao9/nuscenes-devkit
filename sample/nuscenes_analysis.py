from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.nuscenes import NuScenes
import argparse

def analysis_condition():
    eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
        'v1.0-test': 'test'
    }

    eval_version = 'detection_cvpr_2019'
    eval_config = config_factory(eval_version)

    # init the NuScenes dataset
    nusc = NuScenes(version=version_, dataroot=dataroot_, verbose=verbose_)
    nusc_eval = NuScenesEval(
        nusc,
        config=eval_config,
        result_path=result_path_,
        eval_set=eval_set_map[version_],
        output_dir=output_dir_,
        verbose=verbose_,
    )

    # conditional analysis
    distance_gap = 10
    for i in range (5):
        min_distance = i * distance_gap
        max_distance = (i + 1) * distance_gap
        filter_condition = {"distance": [min_distance, max_distance]}

        print('\n\n############filter_condition####################')
        print(filter_condition)
        print('#############################################')
        metrics_summary = nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_, filter_condition=filter_condition)

    speed_gap = 5
    for i in range (5):
        min_speed = i * speed_gap
        max_speed = (i + 1) * speed_gap
        filter_condition = {"speed": [min_speed, max_speed]}

        print('\n\n############filter_condition####################')
        print(filter_condition)
        print('#############################################')
        metrics_summary = nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_, filter_condition=filter_condition)
    metrics_summary = nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)

if __name__ == "__main__":
    
    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='./nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = args.result_path
    output_dir_ = args.output_dir
    dataroot_ = args.dataroot
    version_ = args.version
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    analysis_condition()