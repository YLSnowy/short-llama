# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama.serial_generation import Llama_new
# from llama.predict_length import CNN2D
# from llama.CnnModel import CNN_MaxBSZ
# from llama.generation import Llama_new
from llama.predict_length import ModelArgs,Classifier

# from pyinstrument import Profiler
# from memory_profiler import profile
import json
import torch
import csv
import os

dialogs_origin = []
def read_file(file_path, file_name):
    i = 0
    with open(file_path + file_name, "r") as f:
        for line in f.readlines():
            content = json.loads(line)
            # print(line)

            # print(content['question'])
            # break
            role_content = {}
            role_content["role"] = "user"
            # role_content["content"] = content['question'] # + " option1: " + content['option1'] + " option2: " + content['option2']
            if(role_content["content"] == "exit"):
                break
            
            dialog = []
            dialog.append(role_content)
            dialogs_origin.append(dialog)
    # print(dialogs_origin)
    return dialogs_origin


def read_input():
    while(True):
        role_content = {}
        role_content["role"] = "user"
        role_content["content"] = input("please input your query:\n")
        # role_content["content"] = "what is python"
        if(role_content["content"] == "exit"):
            break
                
        dialog = []
        dialog.append(role_content)
        dialogs_origin.append(dialog)
    return dialogs_origin
    

def read_pth(file_path, file_name):
    data = torch.load(file_path + file_name)
    for i in range(0, len(data)):
        role_content = {}
        role_content["role"] = "user"
        role_content["content"] = data[i]
                
        dialog = []
        dialog.append(role_content)
        dialogs_origin.append(dialog)


def read_triviaqa(file_path, file_name):
    i = 0
    with open(file_path + file_name, "r") as f:
        content = json.loads(f.read())
        # print(type(content["Data"]))
        
        for qa in content["Data"]:
            role_content = {}
            role_content["role"] = "user"
            role_content["content"] = qa['Question']
            
            dialog = []
            dialog.append(role_content)
            dialogs_origin.append(dialog)

            i += 1
            if i == 1600:
                break
    return dialogs_origin


def read_winograde(file_path, file_name):
    i = 0
    with open(file_path + file_name, "r") as f:
        for line in f.readlines():
            content = json.loads(line)
            
            role_content = {}
            role_content["role"] = "user"
            role_content["content"] = content['sentence'] + ' option1: ' + content['option1'] + '. option2: ' + content['option2']

            dialog = []
            dialog.append(role_content)
            dialogs_origin.append(dialog)

            # i += 1
            # if i == 4:
            #     break
    return dialogs_origin



def read_MMLU(file_path, file_name):
    with open(file_path + file_name, "r") as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            role_content = {}
            role_content["role"] = "user"
            role_content["content"] = "Q: " + row[0] + '\n' + ' (A)' + row[1] + '. (B)' + row[2] + '. (C)' + row[3] + '. (D)' + row[4]
            print(role_content)

            dialog = []
            dialog.append(role_content)
            dialogs_origin.append(dialog)
    return dialogs_origin


def read_RACE(file_path):
    for root, dirs, files in os.walk(file_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    content = json.loads(line)
                    for i in range(len(content['questions'])):
                        role_content = {}
                        role_content["role"] = "user"
                        role_content['content'] = content['questions'][i]
                        for j in range(len(content['options'][i])):
                            role_content["content"] += ' option' + str(j+1) + ': ' + content['options'][i][j]
                            
                        dialog = []
                        dialog.append(role_content)
                        dialogs_origin.append(dialog)
                    print(dialogs_origin)
    return dialogs_origin


# @profile
# def main(
#     ckpt_dir: str,
#     tokenizer_path: str,
#     temperature: float = 0.6,
#     top_p: float = 0.9,
#     max_seq_len: int = 512,
#     max_batch_size: int = 1,
#     max_gen_len: Optional[int] = None,
# ):

def main(
    ckpt_dir: str = '/home/snow/llama/llama/llama_params/llama-2-7b-chat/',
    tokenizer_path: str = '/home/snow/llama/llama/tokenizer.model',
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 521,
    max_batch_size: int = 32,
    max_gen_len: Optional[int] = None,
):
    predictor_params = {
        "MODEL": "multichannel",
        "MAX_SENT_LEN": 10,
        "BATCH_SIZE": 50,
        "WORD_DIM": 4096,
        "VOCAB_SIZE": 32000,
        "CLASS_SIZE": 3,
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
    }
    # len_predictor = CNN2D(**predictor_params).to(torch.device('cuda'))
    # len_predictor = CNN_MaxBSZ().to(torch.device('cuda'))
    # len_predictor.load_state_dict(torch.load('/home/snow/llama/zdm_1_0.88973.pkl'))
    # len_predictor = None

    param = ModelArgs(dim=32, n_layers=1, n_heads=8, input_length=80)
    len_predictor = Classifier(param).cuda()
    len_predictor.load_state_dict(torch.load('/home/snow/llama/LlamaEmb_32_1_8_0.6612.pth'))
    # len_predictor = None


    # generator = Llama.build(
    generator = Llama_new.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    i = 0
    answers = []
    while(True):
        dialogs = dialogs_origin[i:i + generator.model.params.max_batch_size]
        # dialogs = dialogs_origin[i:i + 1] # 保证同时执行一批还是执行一个
        if len(dialogs) <= 0:
            break
        i += generator.model.params.max_batch_size
        # i += 1 # 同上
        # print("dialogs is = ", dialogs)
        results = generator.chat_completion(
            len_predictor=len_predictor,
            dialogs=dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        print("batch end==========================")
        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")
            answers.append(result['generation']['content'])
        # torch.save(answers, './result/CNN_MaxBSZ-40-answers.pth')
        # torch.save(answers, './result/triviaqa-llama-100-answers.pth')
        # torch.save(answers, './result/winogrande-llama-30-answers.pth')
        # torch.save(answers, './result/question-llama-answers.pth')


if __name__ == "__main__":
    # profiler = Profiler()
    # profiler.start()
    
    # read_file("./dataset/natural-questions/nq_open/", "NQ-open.dev.jsonl")
    # read_input()
    # read_pth('./dataset/', 'questions.pth')
    read_triviaqa('./dataset/triviaqa-unfiltered/', 'unfiltered-web-dev.json')
    # read_winograde('./dataset/winogrande_1.1/', 'train_l.jsonl')
    # read_MMLU('./dataset/MMLU/test/', 'abstract_algebra_test.csv')
    # read_RACE('./dataset/RACE/test/high')
    # print(dialogs_origin)
    fire.Fire(main)

    # profiler.stop()
    # profiler.print()
    