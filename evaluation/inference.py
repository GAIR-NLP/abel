from vllm import LLM, SamplingParams
import os
import re
import json
import jsonlines
import argparse
import torch
from tqdm import tqdm
import sys
import pdb
from evaluation.math_normalization import *

def get_results(pred_file, dev_set):
    def test_answer(pred_str, ans_str):
        pattern = "#### (.*)$"

        if "Question" in pred_str:
            pred_str = pred_str.split("Question")[0]

        preds = re.findall(pattern, pred_str)
        pred = preds[-1] if len(preds) >= 1 else ""
        if "</s>" in pred:
            pred = pred[:-4]
        
        gold = ans_str
        pred = normalize_final_answer(pred)
        gold = normalize_final_answer(gold)
        return check_sympy_equivalence(gold, pred), pred, gold
    
    def parse_pred_ans(preds_str, golds_str, properties_list):
        num_q = 0
        acc = 0
        results = []
        preds = []
        golds = []
        correct_table = {}
        cnt_table = {}
        source_set = set()
        for pred_str, gold_str, properties in tqdm(zip(preds_str, golds_str, properties_list), total=len(preds_str)):
            num_q += 1
            result, pred, gold = test_answer(pred_str, gold_str)
            results.append(result)
            preds.append(pred)
            golds.append(gold)
            if result:
                acc += 1
            source = properties['source']
            tag = properties['tag']
            source_set.add(source)
            if source not in correct_table.keys():
                correct_table[source] = 1 if result else 0
                cnt_table[source] = 1
            else:
                correct_table[source] = (correct_table[source] + 1) if result else correct_table[source]
                cnt_table[source] += 1
            for key in tag.keys():
                value = tag[key]
                value = source+","+key+"__"+value
                if value not in correct_table.keys():
                    correct_table[value] = 1 if result else 0
                    cnt_table[value] = 1
                else:
                    correct_table[value] = (correct_table[value] + 1) if result else correct_table[value]
                    cnt_table[value] += 1
        print('num_q %d correct %d ratio %.4f' % (num_q, acc, float(acc / num_q)))
        acc_table = {}
        for key in correct_table.keys():
            acc_table[key] = correct_table[key] / cnt_table[key]
        acc_table = list(zip(acc_table.keys(), acc_table.values()))
        acc_table.sort(key=lambda x: x[0])
        for key, acc in acc_table:
            if key in source_set:
                print(key+" : "+str(acc))
            else:
                print("    " + key.split(",")[-1]+ " : " + str(acc))
        return results, preds, golds

    if dev_set in ['all', 'gsm8k', 'math', 'mathgpt', 'gsm8k_robust']:
        golds_str = []
        properties = []
        with open(f'./data/test/test.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if dev_set != "all":
                    if json.loads(line)['source'].lower() == dev_set:
                        golds_str.append(json.loads(line)['target'])
                        properties.append({"source": json.loads(line)['source'], "tag": json.loads(line)['tag']})
                else:
                    golds_str.append(json.loads(line)['target'])
                    properties.append({"source": json.loads(line)['source'], "tag": json.loads(line)['tag']})
        preds_str = []
        with open(pred_file, 'r', encoding='utf-8') as f:
            for line in f:
                preds_str.append(json.loads(line)['response'])
        results, preds, golds = parse_pred_ans(preds_str, golds_str, properties)
        with open(pred_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        for i, line in enumerate(data):
            line['pred'] = preds[i]
            line['gold'] = golds[i]
            line['result'] = results[i]

        # Save the updated list of dictionaries back to the jsonl file
        with open(pred_file, 'w') as file:
            for item in data:
                file.write(json.dumps(item) + '\n')

    else:
        raise NotImplementedError("Evaluation not supported.")


def get_raw_inputs(dev_set):
    # in this function, we will get the raw queries for a target dev set
    data = []
    if dev_set in ['all', 'gsm8k', 'math', 'mathgpt', 'gsm8k_robust']:
        with open(f'./data/test/test.jsonl') as f:
            for line in jsonlines.Reader(f):
                data.append(line)
        if dev_set != 'all':
            data = [line for line in data if line['source'].lower() == dev_set]
    else:
        raise ValueError

    prompt_list = [line['question'] for line in data]
    return prompt_list


prompt_mapping = {
    "math-single": "Question:\n{input}\nAnswer:\nLet's think step by step.\n",
}

if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--output_file_name', type=str, default='output.json')
    parser.add_argument('--stop', type=str, nargs='+', default=[], help="you can pass one or multiple stop strings to halt the generation process.")
    parser.add_argument('--dev_set', type=str, default='all')
    parser.add_argument('--prompt_type', type=str, default='math-single')
    parser.add_argument('--sample_num', type=int, default=-1, )
    parser.add_argument('--eval_only', type=bool, default=False)
    parser.add_argument('--max_num_batched_tokens', type=int, default=2048)
    args = parser.parse_args()

    if args.eval_only == False:
        # part 1 we set the model
        num_gpus = torch.cuda.device_count()
        another_args = {'max_num_batched_tokens': args.max_num_batched_tokens} 
        llm = LLM(model=args.model_dir, tensor_parallel_size=num_gpus,**another_args)
        print('>>>>>> model loaded')
        # part 2 we set the sampling params
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                                            stop=args.stop, presence_penalty=args.presence_penalty,
                                            frequency_penalty=args.frequency_penalty)

        # part 3 we prepare raw queries and wrap them with target prompt
        raw_queries = get_raw_inputs(args.dev_set)
        prompt = prompt_mapping[args.prompt_type]
        processed_prompts = [prompt.format(input=query) for query in raw_queries]
        processed_prompts = processed_prompts[:args.sample_num] if args.sample_num > 0 else processed_prompts

        # part 4 we generate, note that vllm is async so extra sorting is needed
        outputs = llm.generate(processed_prompts, sampling_params)
        sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
        print('>>>>>> generation done')

        # part 5 we save the results, always be {'id':id,'response':response}
        # if dir of output file is not exist, it will be created automatically
        if not os.path.exists(os.path.dirname(args.output_file_name)):
            os.makedirs(os.path.dirname(args.output_file_name))
        with open(args.output_file_name, "w") as f:
            for id, output in enumerate(sorted_outputs):
            # note that `prompt`s are the wrapped ones
                f.write(json.dumps({'id': id, 'prompt': output.prompt, 'response': output.outputs[0].text}) + '\n')
        print('>>>>>> writing prediction done')

    # part 6 evaluate, I guess this should be done in a separate script
    get_results(args.output_file_name, args.dev_set)
    print('>>>>>> evaluation done')
