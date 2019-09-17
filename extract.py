#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from tqdm import tqdm

from model_loader import load_checkpoint, make_model
from youcook2.dataset import Youcook2DataSet
from ops.transforms import *

parser = argparse.ArgumentParser(
    description="Test the instantiation and forward pass of models",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "model_type",
    nargs="?",
    choices=["tsn", "tsm", "tsm-nl", "trn", "mtrn"],
    default=None,
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    help="Path to checkpointed model. Should be a dictionary containing the keys:"
    " 'model_type', 'segment_count', 'modality', 'state_dict', and 'arch'.",
)
parser.add_argument(
    "--arch",
    default="resnet50",
    choices=["BNInception", "resnet50"],
    help="Backbone architecture",
)
parser.add_argument(
    "--modality", default="RGB", choices=["RGB", "Flow"], help="Input modality"
)
parser.add_argument(
    "--flow-length", default=5, type=int, help="Number of (u, v) pairs in flow stack"
)
parser.add_argument(
    "--dropout",
    default=0.7,
    type=float,
    help="Dropout probability. The dropout layer replaces the "
    "backbone's classification layer.",
)
parser.add_argument(
    "--trn-img-feature-dim",
    default=256,
    type=int,
    help="Number of dimensions for the output of backbone network. "
    "This is effectively the image feature dimensionality.",
)
parser.add_argument(
    "--segment-count",
    default=8,
    type=int,
    help="Number of segments. For RGB this corresponds to number of "
    "frames, whereas for Flow, it is the number of points from "
    "which a stack of (u, v) frames are sampled.",
)
parser.add_argument(
    "--tsn-consensus-type",
    choices=["avg", "max"],
    default="avg",
    help="Consensus function for TSN used to fuse class scores from "
    "each segment's predictoin.",
)
parser.add_argument(
    "--tsm-shift-div",
    default=8,
    type=int,
    help="Reciprocal proportion of features temporally-shifted.",
)
parser.add_argument(
    "--tsm-shift-place",
    default="blockres",
    choices=["block", "blockres"],
    help="Location for the temporal shift to take place. Either 'block' for the shift "
    "to happen in the non-residual part of a block, or 'blockres' if the shift happens "
    "in the residual path.",
)
parser.add_argument(
    "--tsm-temporal-pool",
    action="store_true",
    help="Gradually temporally pool throughout the network",
)
parser.add_argument("--batch-size", default=16, type=int, help="Batch size for demo")
parser.add_argument("--print-model", action="store_true", help="Print model definition")

parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--root_path', default="/s1_md0/v-fanxu/Extraction/youcook2",
                    help='path of dataset')
parser.add_argument('--infer_list', default="/s1_md0/v-fanxu/Extraction/youcook2/manifest.txt",
                    help='path of dataset')
parser.add_argument('--verb_list', default="/s1_md0/v-fanxu/junyidu/github/action-models/EPIC_verb_classes.csv",
                    help='path of dataset')

def extract_settings_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    settings = vars(args)
    for variant in ["trn", "tsm", "tsn"]:
        variant_key_prefix = f"{variant}_"
        variant_keys = {
            key for key in settings.keys() if key.startswith(variant_key_prefix)
        }
        for key in variant_keys:
            stripped_key = key[len(variant_key_prefix) :]
            settings[stripped_key] = settings[key]
            del settings[key]
    return settings


def main(args):
    logging.basicConfig(level=logging.INFO)
    if args.checkpoint is None:
        if args.model_type is None:
            print("If not providing a checkpoint, you must specify model_type")
            sys.exit(1)
        settings = extract_settings_from_args(args)
        model = make_model(settings)
    elif args.checkpoint is not None and args.checkpoint.exists():
        model = load_checkpoint(args.checkpoint)
    else:
        print(f"{args.checkpoint} doesn't exist")
        sys.exit(1)

    if args.print_model:
        print(model)
    height, width = model.input_size, model.input_size
    if model.modality == "RGB":
        channel_dim = 3
        data_length = 1
    elif model.modality == "Flow":
        channel_dim = args.flow_length * 2
        data_length = 5
    else:
        raise ValueError(f"Unknown modality {args.modality}")

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std

    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    model = torch.nn.DataParallel(model).cuda()

    val_loader = torch.utils.data.DataLoader(
        Youcook2DataSet(args.root_path, args.infer_list, num_segments=args.segment_count,
                   new_length=data_length,
                   modality=args.modality,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    verb_logits, noun_logits = inference(val_loader, model)
    np.save(open('logits_verb.npy', 'wb'), verb_logits)
    np.save(open('logits_noun.npy', 'wb'), noun_logits)
    predict(args.verb_list, verb_logits, 'verb')
    predict(args.noun_list, noun_logits, 'noun')

def inference(val_loader, model):
    # switch to evaluate mode
    print("Validation")
    model.eval()

    verb_logits_lst = list()
    noun_logits_lst = list()
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(val_loader)):
            # compute output
            output = model(input)
            verb_logits_lst.append(output[0].cpu().numpy())
            noun_logits_lst.append(output[1].cpu().numpy())

    verb_logits = np.vstack(verb_logits_lst)
    noun_logits = np.vstack(noun_logits_lst)

    return verb_logits, noun_logits

def predict(annotation, output, name):
    predicted = np.argmax(output, axis=1)
    predicted = predicted.astype('int')

    rows = [line.strip().split(',') for line in open(annotation, 'r')]
    id2key = {int(row[0]):row[1] for row in rows[1:]}
    
    keys = [id2key[id] for id in predicted]
    with open(f'predict_{name}.txt', 'w') as f:
        f.write('\n'.join(keys))

if __name__ == "__main__":
    main(parser.parse_args())
