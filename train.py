import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim
import json
import pandas as pd
import csv

# HyperParameter
batch_size = 32
learning_rate = 5e-5
log_iter = 10
epochs = 5


# Load Data
data_path = 'all_evallog.jsonl'
train_data = []
REWARD_NAME=['enjoy', 'interest', 'listen', 'turing', 'avoid_rep', 'make_sense', 'fluency', 'persona_guess','inquisitive', 'reward']

with open(data_path,'r') as datafile:
    for l in datafile:
        sample = json.loads(l.strip())
        dialog = sample['dialog']
        dialog = [turn['text'] for turn in dialog]
        sample['evaluation_results']['reward'] = sum(sample['evaluation_results'].values())/9.
        reward = [sample['evaluation_results'][rn]/5. for rn in REWARD_NAME]
        train_data.append({'dialog': " ".join(dialog), 'reward': reward})
# train_data = torch.utils.data.Dataset(train_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Load Eval Data
human_eval = pd.read_csv('stats.csv')
human_eval = human_eval[human_eval['model_name']!='human_eval']
for rn in REWARD_NAME:
    human_eval[rn] = human_eval.pop(rn+'-mean')
    human_eval.pop(rn+'-std')
self_play_log_path = 'selfplay'
self_play_data = {}
sample_data = None
for model_name in human_eval['model_name'][:]:
    try:
        single_data = [json.loads(l) for l in open(self_play_log_path+'/'+model_name+'.jsonl')][0]
        single_data = [" ".join([" ".join(t) for t in l]) for l in single_data]
        sample_data = single_data
    except:
        single_data = sample_data
    self_play_data[model_name] = single_data

# Load Model
model_name = "roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=len(REWARD_NAME))
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
@torch.no_grad()
def eval(model):
    model.eval()
    eval_df = human_eval.copy()
    all_pearson_r = {}
    all_spearman_r = {}
    for rn in REWARD_NAME:
        eval_df[rn+'_sp'] = 0
    for model_name in human_eval['model_name'][:]:
        pt_batch = tokenizer(
            self_play_data[model_name],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        input_ids = pt_batch['input_ids'].cuda()
        attention_mask = pt_batch['attention_mask'].cuda()
        output = model(input_ids=input_ids,attention_mask=attention_mask)['logits']
        ests = output.sum(0)
        for rn, est in zip(REWARD_NAME, ests):
            eval_df.loc[eval_df['model_name']==model_name,rn+"_sp"] = est.item()

    for rn in REWARD_NAME:
        select_df1 = eval_df[rn]
        select_df2 = eval_df[rn+"_sp"]
        pearson_r = select_df1.corr(select_df2)
        spearman_r = select_df1.corr(select_df2,method='spearman')
        all_pearson_r[rn] = pearson_r
        all_spearman_r[rn] = spearman_r
        print(f"{rn}: {pearson_r:.4f} {spearman_r:.4f}")
    
    return all_pearson_r, all_spearman_r

# eval(model)
# exit()

# Training
def train(model, dataloader, optimizer):
    lossfun = nn.MSELoss()
    iter = 0
    total_iter = len(dataloader)*epochs
    best_score = -100
    best_result = None

    for ep in range(epochs):
        for batch in dataloader:
            model.train()
            iter += 1
            text = batch['dialog']
            reward = torch.stack(batch['reward'], 1).float().cuda()
            # print(reward.shape)
            # reward = batch['reward'].cuda()
            pt_batch = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            input_ids = pt_batch['input_ids'].cuda()
            attention_mask = pt_batch['attention_mask'].cuda()
            
            optimizer.zero_grad()
            output = model(input_ids=input_ids,attention_mask=attention_mask)['logits']
            loss = lossfun(output, reward)
            loss.backward()
            optimizer.step()
            if iter%log_iter == 0:
                print(f"[{iter}/{total_iter}] loss: {loss.item()}")
            # break
        result = eval(model)
        avg_score = sum(list(result[1].values()))
        if avg_score > best_score:
            best_score = avg_score
            best_result = result
            print("New Best: ")
            for rn in REWARD_NAME:
                print(f'{rn:>15} {best_result[0][rn]:.4f}  {best_result[1][rn]:.4f}')
            # print(f"best_result: {best_result}")
        print(f"Evaluation avg_score: {avg_score: .4f}, best_score: {best_score: .4f}")
    return best_result

best_result = train(model, train_loader, optimizer)
with open('best_result.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    employee_writer.writerow(["Metric"]+REWARD_NAME)
    employee_writer.writerow(["Pearson"]+ [best_result[0][rn] for rn in REWARD_NAME])
    employee_writer.writerow(["Spearman"]+ [best_result[1][rn] for rn in REWARD_NAME])