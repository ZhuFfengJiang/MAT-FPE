import os

from evaluator.coco_evaluator import COCOAPIEvaluator




def build_evluator(args, data_cfg, transform, device):
    # Basic parameters
    data_dir = os.path.join(args.root, data_cfg['data_name'])

    ## COCO Evaluator
    if args.dataset == 'coco':
        evaluator = COCOAPIEvaluator(data_dir  = data_dir,
                                     device    = device,
                                     transform = transform
                                     )

    return evaluator
