import torch
import json
from bert_score import score
# from bleurt import score
import matplotlib.pyplot as plt
from bert_score import plot_example
from collections import defaultdict
import evaluate
import datasets
from numpy import *


def print_answer_length(answers1:list, answers2:list):
    percent = []
    for i in range(len(answers1)):
        tmp = (float)(len(answers1[i])) / (float)(len(answers2[i]))
        percent.append(tmp)
    print("prompt percent = ", mean(percent))
    statictis = [0 for i in range(0, 11)]
    for i in range(len(percent)):
        if percent[i] < 0.1:
            statictis[0] += 1
        if percent[i] < 0.2:
            statictis[1] += 1
        if percent[i] < 0.3:
            statictis[2] += 1
        if percent[i] < 0.4:
            statictis[3] += 1
        if percent[i] < 0.5:
            statictis[4] += 1
        if percent[i] < 0.6:
            statictis[5] += 1
        if percent[i] < 0.7:
            statictis[6] += 1
        if percent[i] < 0.8:
            statictis[7] += 1
        if percent[i] < 0.9:
            statictis[8] += 1
        if percent[i] < 1.0:
            statictis[9] += 1
        if percent[i] >= 1.0:
            statictis[10] += 1
    print(statictis)
    return statictis


def read_json(length:int):
    i = 0
    answers_standard = []
    with open('../dataset/NQ-open.train.jsonl', "r") as f:
        for line in f.readlines():
            content = json.loads(line)

            answers_standard.append(content['answer'][0])
            i += 1
            if i == length:
                break
    return answers_standard


def cal_score(answer1, answer2):
    # P, R, F1 = score(answer1, answer2,model_type="roberta-large",lang="en", verbose=True)
    # print(F1)
    # P, R, F1 = score(answer1, answer2,model_type="roberta-large",lang="en", verbose=True)
    # print(F1)
    P, R, F1 = score(answer1, answer2,model_type="bert-base-chinese",lang="en", verbose=True)
    print(F1)
    # P, R, F1 = score(answer1, answer2,model_type="microsoft/deberta-large-mnli",lang="en", verbose=True)
    # print(F1)
    # P, R, F1 = score(answer1, answer2,model_type="microsoft/deberta-xlarge-mnli",lang="en", verbose=True)
    # print(F1)
    # P, R, F1 = score(answer1, answer2,model_type="facebook/bart-large-mnli",lang="en", verbose=True)
    # print(F1)
    
    print(f"System level F1 score: {F1.mean():.3f}")


def cal_perplexity(answers:list):
    print(len(answers))
    perplexity = evaluate.load("./evaluate/metrics/perplexity", module_type="metric")
    answers = ["lorem ipsum", "Happy Birthday!", "how are you"]
    results = perplexity.compute(model_id='roberta-large',add_start_token=False,predictions=answers)
    print(results)


def cal_blurt(answer1, asnwer2):
    checkpoint = "./BLEURT-20"
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=answer1, candidates=asnwer2)
    print(scores)


def gene_json(answer1:list, answer2:list, n:int):
    res = []
    question = torch.load('../dataset/questions.pth')
    for i in range(100):
        mid_res = {}
        mid_res["question_id"] = i
        mid_res["question"] = question[i]
        mid_res["response"] = {}
        mid_res["response"]["llama"] = answer1[i]
        mid_res["response"]["llama-short"] = answer2[i]
        res.append(mid_res)

    with open("./test-" + str(n) + ".json", "w") as f:
        json.dump(res, f, indent=4)


def cal_wino_accu(answers):
    standard_answers = []
    with open("../dataset/winogrande_1.1/train_l.jsonl", "r") as f:
        for line in f.readlines():
            content = json.loads(line)
            standard_answers.append(content['answer'])
    
    count = 0
    accu = 0
    for i in range(len(answers)):
        if 'answer is' in answers[i]:
            count += 1
            place = answers[i].find('answer is')
            if 'option 1' in answers[i][place:place+20] or 'option1' in answers[i][place:place+20] or '(a)' in answers[i][place:place+20] or '(A)' in answers[i][place:place+20] and standard_answers[i] == '1':
                accu += 1
            if 'option 2' in answers[i][place:place+20] or 'option2' in answers[i][place:place+20] or '(b)' in answers[i][place:place+20] or '(B)' in answers[i][place:place+20] and standard_answers[i] == '2':
                accu += 1

    print(count)
    print(accu)


def get_avg_time(file_name, n):
    time = [0 for _ in range(50)]
    i = 0
    count = 1
    with open(file_name, "r") as f:
        for line in f.readlines():
            if len(line.split(' ')) > 1:
                time[i] += (float)(line.strip().split(' ')[-1])
                i += 1
            else:
                i = 0
                count += 1
    for i in range(n):
        time[i] /= count
    print(time)
    return time


if __name__ == "__main__":
    # answers_10 = torch.load('./triviaqa-llama-40-answers.pth')
    # answers_20 = torch.load('./final_result/20-answers.pth')
    # answers_30 = torch.load('./final_result/30-answers.pth')
    # answers_40 = torch.load('./final_result/40-answers.pth')
    # answers_50 = torch.load('./final_result/50-answer.pth')
    # answers_80 = torch.load('./final_result/80-answers.pth')
    # answers_100 = torch.load('./final_result/100-answers.pth')
    # answers_200 = torch.load('./final_result/200-answer.pth')
    # answers = torch.load('./question-llama-answers.pth')
    # answers_wino_40 = torch.load('./winogrande-llama-40-answers.pth')
    # answers_wino = torch.load('./winogrande-llama-answers.pth')

    # answers_tri_10 = torch.load('./triviaqa-llama-10-answers.pth')
    # answers_tri = torch.load('./triviaqa-llama-answers.pth')
    # print(len(answers_tri))

    # cal_wino_accu(answers_wino)
    # cal_wino_accu(answers_wino_40)
    
    # length = len(answers_tri_200)
    # answers_standard = read_json(length)
    # answers_standard = [answers_standard[i] for i in range(length)]

    # print_answer_length(answers_tri_10, answers_tri)
    # print_answer_length(answers_20, answers)
    # print_answer_length(answers_30, answers)
    # print_answer_length(answers_40, answers)
    # print_answer_length(answers_50, answers)
    # print_answer_length(answers_80, answers)
    # print_answer_length(answers_100, answers)
    # print_answer_length(answers_200, answers)

    # gene_json(answers, answers_200, 200)


    # cal_score(['hi', 'In the Attack on Titan series, Annie Leonhart is a member of the Scouting Legion and works directly under the command of the Military Police Brigade. She does not work directly for any specific character or organization, but rather serves as a member of the military forces that are fighting against the Titans.'], ['hello', 'Marley'])
    # cal_score(answers_tri_200, answers_standard)
    # cal_score(answers_tri_200, answers_tri)
    # cal_score(answers, answers_standard)

    # cal_perplexity(answers_30)
    # cal_perplexity(answers)
    # cal_perplexity(answers_standard)

    # cal_blurt(['hi', 'balabala'], ['hello', 'what animal is mr. big in the movie zootopia'])
    # cal_blurt(['hi', 'In the Attack on Titan series, Annie Leonhart is a member of the Scouting Legion and works directly under the command of the Military Police Brigade. She does not work directly for any specific character or organization, but rather serves as a member of the military forces that are fighting against the Titans.'], ['hello', 'Marley'])
    # cal_blurt(answers_30, answers)
            
    get_avg_time('tri-llama-time.txt', 50)
    get_avg_time('tri-llama-short-time.txt', 50)