import os
import re
import argparse
import json
import time

from openai import OpenAI


# template = '''You are required to score the performance of two AI assistants in describing a given image. 
# You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image or describing incorrectly in terms of the counts, positions, or colors of objects in the image. 

# Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria: 
# 1: Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinations should be given higher scores. 
# 2: Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not count as necessary details. 

# Please output the scores for each criterion, containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. 

# [Question]
# {}
# [End of Question]

# [Assistant 1] 
# {} 
# [End of Assistant 1] 

# [Assistant 2] 
# {} 
# [End of Assistant 2] 

# Output format: 
# Accuracy: 
# Reason: 
# Detailedness: 
# Reason:
# '''

template = '''You are required to score the performance of two AI assistants in describing a given image. 
You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image or describing incorrectly in terms of the counts, positions, or colors of objects in the image. 

Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria: 
1: Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinations should be given higher scores. 
2: Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not count as necessary details. 
3: Naturalness: assess the language quality, focusing on: fluency of sentence structure, appropriateness of word choice, smoothness of language flow, absence of awkward or unnatural phrasing.

Please output the scores for each criterion, containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. 

[Question]
{}
[End of Question]

[Assistant 1] 
{} 
[End of Assistant 1] 

[Assistant 2] 
{} 
[End of Assistant 2] 

Output format:
Accuracy:
Reason:
Detailedness: 
Reason:
Naturalness:
Reason:
'''

def scores_each_type(types, scores):
    scores_dict = {}
    for i, score in enumerate(scores):
        if types[i] not in scores_dict:
            scores_dict[types[i]] = []
        scores_dict[types[i]].append(score)
    scores_dict = {t: sum(s)/len(s) for t, s in scores_dict.items()}
    
    return scores_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, default='data/llava-bench-in-the-wild/questions.jsonl', help='question file')
    parser.add_argument('--response1', type=str, default='', help='response file containing images, questions, and model responses')
    parser.add_argument('--response2', type=str, default='', help='response file containing images, questions, and model responses')
    parser.add_argument('--api-key', type=str, default="")
    parser.add_argument('--gpt-model', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.evaluation = os.path.join(os.path.dirname(args.response2), 
                                   args.gpt_model + '-' + str(args.seed) + '-' + 
                                   os.path.basename(args.response1).split('-')[1] + os.path.basename(args.response2))

    client = OpenAI(api_key=args.api_key, base_url="")
    images_url = "https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/images/"
    questions = [json.loads(l) for l in open(args.question, 'r')]

    if not os.path.exists(args.evaluation):

        # load json file
        answers1 = [json.loads(l) for l in open(args.response1, 'r')]
        answers2 = [json.loads(l) for l in open(args.response2, 'r')]

        assert len(answers1) == 60
        assert len(answers2) == 60

        # ask GPT-4 to evaluate
        responses = []
        for i, (answer1, answer2) in enumerate(zip(answers1, answers2)):
            question = questions[i]
            assert answer1['question_id'] == answer2['question_id'] == question['question_id']
            input_text = template.format(answer1['question'], answer1['text'], answer2['text'])
            image_url = images_url + question['image']
            # print(input_text)

            response = None
            while response is None:
                try:
                    response = client.chat.completions.create(
                        model=args.gpt_model,
                        messages=[
                            {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": input_text},
                                {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                                },
                            ],
                            }
                        ],
                        # max_tokens=300,
                        )
                except Exception as e:
                    print(e)
                    print('retrying...')
                    time.sleep(20)
                    continue
            
            response = response.choices[0].message.content
            print(i, response, flush=True)
            responses.append(response)
            time.sleep(1)

        # save responses
        with open(args.evaluation, 'w') as f:
            json.dump(responses, f, indent=2)
    
    else:
        print("Loading existing evaluation results from", args.evaluation)
        responses = json.load(open(args.evaluation, 'r'))

    acc_pattern = re.compile(r'Accuracy:(.*?)(\d+?)(.+?)(\d+?)')
    det_pattern = re.compile(r'Detailedness:(.*?)(\d+?)(.+?)(\d+?)')
    nat_pattern = re.compile(r'Naturalness:(.*?)(\d+?)(.+?)(\d+?)')
    # analyze responses
    accs1, accs2, dets1, dets2, nats1, nats2 = [], [], [], [], [], []
    for i, response in enumerate(responses):
        response = response.replace('\n', ' ').replace("**", "").replace("Assistant 1: ", "").replace("Assistant 2: ", "")
        
        try:
            accs = acc_pattern.search(response).groups()
            acc1, acc2 = accs[1], accs[3]
        except:
            acc1, acc2 = 5, 5
            print(response)

        try:
            dets = det_pattern.search(response).groups()
            det1, det2 = dets[1], dets[3]
        except:
            de1, det2 = 5, 5
            print(response)
        
        try:
            nats = nat_pattern.search(response).groups()
            nat1, nat2 = nats[1], nats[3]
        except:
            nat1, nat2 = 5, 5
            print(response)
        
        accs1.append(int(acc1))
        accs2.append(int(acc2))
        dets1.append(int(det1))
        dets2.append(int(det2))
        nats1.append(int(nat1))
        nats2.append(int(nat2))

    types = [q['category'] for q in questions]
    accs1_each = scores_each_type(types, accs1)
    accs2_each = scores_each_type(types, accs2)
    dets1_each = scores_each_type(types, dets1)
    dets2_each = scores_each_type(types, dets2)
    nats1_each = scores_each_type(types, nats1)
    nats2_each = scores_each_type(types, nats2)

    print(f"\nResults for {args.response1}\n")
    print(f"Accuracy:\n{sum(accs1)/len(accs1):.3f}\nDetailedness:\n{sum(dets1)/len(dets1):.3f}\nNaturalness:\n{sum(nats1)/len(nats1):.3f}")
    print(f"Accuracy for each type:\n{accs1_each}\nDetailedness for each type:\n{dets1_each}\nNaturalness for each type:\n{nats1_each}")

    print(f"\nResults for {args.response2}:\n")
    print(f"Accuracy:\n{sum(accs2)/len(accs2):.3f}\nDetailedness:\n{sum(dets2)/len(dets2):.3f}\nNaturalness:\n{sum(nats2)/len(nats2):.3f}")
    print(f"Accuracy for each type:\n{accs2_each}\nDetailedness for each type:\n{dets2_each}\nNaturalness for each type:\n{nats2_each}")
