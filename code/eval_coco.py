from __future__ import print_function

import argparse

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def eval_coco(annFile, resFile):

    coco = COCO(annFile)
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    Bleu_4 = cocoEval.eval['Bleu_4']
    METEOR = cocoEval.eval['METEOR']
    ROUGE_L = cocoEval.eval['ROUGE_L']
    CIDEr = cocoEval.eval['CIDEr']
    total = Bleu_4 + METEOR + ROUGE_L + CIDEr
    score = {'Bleu_4': Bleu_4, 'METEOR': METEOR, 'ROUGE_L': ROUGE_L, 'CIDEr': CIDEr, 'total': total}

    return score


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description="Evaulation of video description")
    argparser.add_argument("ann_json", type=str, help="Json file")
    argparser.add_argument("res_json", type=str, help="Json file")

    args = argparser.parse_args()
    print(eval_coco(args.ann_json, args.res_json))
