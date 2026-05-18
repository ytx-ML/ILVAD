import os
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from pycocotools.coco import COCO
from cocoeval import COCOEvalCap

from chair import CHAIR, load_generated_captions, print_metrics


def eval_chair(answer_file):
    """
    answers: json file: list[dict], keys: "caption", "image_id"
    """
    anno_file = "data/coco/annotations/captions_val2014.json"
    anno_dir = "data/coco/annotations"
    chair_file = os.path.join(os.path.dirname(answer_file), os.path.basename(answer_file).replace('chair', 'chairmetrics'))
    
    try:
        cap_dict = json.load(open(chair_file, 'r'))
        print_metrics(cap_dict)
    except:
        answers = []
        for line in open(answer_file, 'r'):
            #print(line)
            answer = json.loads(line)
            answer['caption'] = answer['text']
            answer['image_id'] = answer['question_id']
            answers.append(answer)
        
        coco = COCO(anno_file)
        formulated_output_dict = {}
        all_overall_scores = defaultdict(list)
        img_to_eval_dict = {}
        chunk_size = 100

        # to save memory, load chunk_size captions at a time
        for s in tqdm(range(0, len(answers), chunk_size)):

            coco_res = coco.loadRes(answers[s: min(s+chunk_size, len(answers))])
            coco_eval = COCOEvalCap(coco, coco_res)
            
            coco_eval.params["image_id"] = coco_res.getImgIds()
            coco_eval.evaluate()

            for metric, score in coco_eval.eval.items():
                all_overall_scores[metric].append(score)
            
            for i, cur_img_id in enumerate(coco_res.getImgIds()):
                cur_eval_dict = coco_eval.evalImgs[i]
                # add caption to the eval dict
                cur_eval_dict["caption"] = coco_res.imgToAnns[cur_img_id][0]["caption"]
                img_to_eval_dict[cur_img_id] = cur_eval_dict

        # overall result
        overall_dict = {}
        for metric, score in all_overall_scores.items():
            overall_dict[metric] = np.mean(score)
        formulated_output_dict["overall"] = overall_dict
        formulated_output_dict["imgToEval"] = img_to_eval_dict

        json.dump(formulated_output_dict, open(chair_file, "w"))

        _, imids, _ = load_generated_captions(chair_file)

        evaluator = CHAIR(imids, anno_dir)
        evaluator.get_annotations()
        cap_dict = evaluator.compute_chair(chair_file)
        json.dump(cap_dict, open(chair_file, "w"))

        print_metrics(cap_dict)


def eval_pope(answers, labels, question_ids):
    pos_num = 0
    pred_list, label_list, error_id = [], [], []
    for question_id in question_ids:
        ### process answer
        if question_id not in answers.keys(): continue
        text = answers[question_id]

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            pred_list.append(0)
        else:
            pred_list.append(1)

        ### process label
        if labels[question_id] and 'no' in labels[question_id].lower():
            label_list.append(0)
        else:
            label_list.append(1)
            pos_num += 1

        ## sta_error
        if pred_list[-1] != label_list[-1]:
            error_id.append(question_id)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    assert len(pred_list) == len(label_list)
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    precision = float(TP) / float(TP + FP + 0.00001)
    recall = float(TP) / float(TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return [acc, precision, recall, f1, yes_ratio]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", type=str)
    parser.add_argument("--eval_chair", action='store_true', default=False)
    parser.add_argument("--eval_pope", action='store_true', default=False)
    args = parser.parse_args()

    if args.eval_chair:

        all_leng = []
        all_chairs = []
        all_chairi = []

        for result_file in glob.glob(args.result_file):
            print(f"Evaluating {result_file} ...")
            eval_chair(result_file)
            subset = result_file.replace('chair', 'chairmetrics')
            data = json.load(open(subset, 'r'))
            sents = data['sentences']
            metrics = data['overall_metrics']

            leng = [len(s['caption'].split()) for s in sents]
            leng = sum(leng) / len(leng)

            all_leng.append(leng)
            all_chairs.append(metrics['CHAIRs'] * 100)
            all_chairi.append(metrics['CHAIRi'] * 100)
        
        print("------------------------------------")
        print("CHAIRs\tstd\tCHAIRi\tstd\tLen")
        print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (
            float(np.mean(all_chairs)), 
            float(np.std(all_chairs)),
            float(np.mean(all_chairi)),
            float(np.std(all_chairi)),
            float(np.mean(all_leng)),
        ))

    elif args.eval_pope:
        avg_results = 0
        for result_file in glob.glob(args.result_file):
            print(f"Evaluating {result_file} ...")
            answers = [json.loads(d) for d in open(result_file, "r")]
            answers_list = {a['question_id']: a['text'] for a in answers}
            label_list = {a['question_id']: a['gt'] for a in answers}
            question_ids = list(label_list.keys())
            result = eval_pope(answers_list, label_list, question_ids)
            results = np.round(np.multiply(np.array(result), 100), decimals=2)
            print("Acc\tPrec\tRec\tF1\tYesRate")
            print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % tuple(results))
            avg_results += results / 3
        print("------------------------------------")
        print("Acc\tPrec\tRec\tF1\tYesRate")
        print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % tuple(avg_results))

        # 添加调试信息
        print("avg_results:", avg_results)
        print("type(avg_results):", type(avg_results))

        # 确保avg_results是一个可迭代对象
        if not isinstance(avg_results, (list, tuple)):
            avg_results = [avg_results]

        # 再次尝试打印
        print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % tuple(avg_results))

    print("====================================")
