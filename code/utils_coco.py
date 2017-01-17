"""
Convert input json to coco format to use COCO eval toolkit
"""

from __future__ import print_function
import argparse
import json
import utils

# remove non-accii characters


def remove_nonaccii(s):
    s = ''.join([i if ord(i) < 128 else '' for i in s])
    return s


def convert(input_txt, output_json):
    infos = utils.read_mapping(input_txt)
    videos = utils.read_keys(input_txt)

    # imgs = [{'id': v} for v in videos]

    anns = [{'caption': remove_nonaccii(' '.join(infos[v])),
             'image_id': v}
            for v in videos]

    out = anns

    with open(output_json, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description="Prepare image input in Json format for evaluation")
    argparser.add_argument("input_txt", type=str, help="Text file")
    argparser.add_argument("output_json", type=str, help="Json file")

    args = argparser.parse_args()

    convert(args.input_txt, args.output_json)
