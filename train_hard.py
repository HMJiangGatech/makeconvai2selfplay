import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim
import json
import pandas as pd
import csv
import os
import numpy as np

# HyperParameter
batch_size = 32
learning_rate = 1e-4
log_iter = 10
epochs = 20
torch.manual_seed(0)
np.random.seed(0)
bert_name = "roberta-base"
REWARD_NAME=['enjoy', 'interest', 'listen', 'turing', 'avoid_rep', 'make_sense', 'fluency', 'persona_guess','inquisitive', 'reward']

# Load Eval Data
human_eval = pd.read_csv('stats.csv')
human_eval = human_eval[human_eval['model_name']!='human_eval']
for rn in REWARD_NAME:
    human_eval[rn] = human_eval.pop(rn+'-mean')
    human_eval.pop(rn+'-std')
self_play_log_path = 'selfplay'
self_play_data = {}
for model_name in human_eval['model_name'][:]:
    single_data = [json.loads(l) for l in open(self_play_log_path+'/'+model_name+'.jsonl')][0]
    single_data = [" ".join([" ".join(t) for t in l]) for l in single_data]
    self_play_data[model_name] = single_data


# Training
def train(model, dataloader, optimizer, target_agent):
    lossfun = nn.MSELoss()
    iter = 0
    total_iter = len(dataloader)*epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iter)
    scaler = torch.cuda.amp.GradScaler()

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
                max_length=448,
                return_tensors="pt"
            )
            input_ids = pt_batch['input_ids'].cuda()
            attention_mask = pt_batch['attention_mask'].cuda()
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(input_ids=input_ids,attention_mask=attention_mask)['logits']
                loss = lossfun(output, reward)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if iter%log_iter == 0:
                print(f"[{iter}/{total_iter}] loss: {loss.item()}")
            # break
    result = eval(model, target_agent)
    return result

# Testing
@torch.no_grad()
def eval(model, target_agent):
    model.eval()
    pt_batch = tokenizer(
        self_play_data[target_agent],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = pt_batch['input_ids'].cuda()
    attention_mask = pt_batch['attention_mask'].cuda()
    output = model(input_ids=input_ids,attention_mask=attention_mask)['logits']
    result = output.sum(0)
    
    return result

for rn in REWARD_NAME:
    human_eval[rn+'_sp'] = 0
for agent_id, target_agent in enumerate(human_eval['model_name'][:]):
    print(f"======={agent_id}======{target_agent}")
    # Load Data
    train_data = []
    data_files = os.listdir('evallog')
    for f in data_files:
        if f.split('.')[0] == target_agent:
            continue
        data_path = 'evallog/'+f
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
    print(len(train_data))

    # Load Model
    model = AutoModelForSequenceClassification.from_pretrained(bert_name,num_labels=len(REWARD_NAME))
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    results = train(model, train_loader, optimizer, target_agent)
    
    for rn, est in zip(REWARD_NAME, results):
        human_eval.loc[human_eval['model_name']==target_agent,rn+"_sp"] = est.item()

    all_pearson_r = {}
    all_spearman_r = {}
    for rn in REWARD_NAME:
        select_df1 = human_eval[rn]
        select_df2 = human_eval[rn+"_sp"]
        pearson_r = select_df1.corr(select_df2)
        spearman_r = select_df1.corr(select_df2,method='spearman')
        all_pearson_r[rn] = pearson_r
        all_spearman_r[rn] = spearman_r
        print(f"{rn}: {pearson_r:.4f} {spearman_r:.4f}")
    
with open('last_result_hard.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    employee_writer.writerow(["Metric"]+REWARD_NAME)
    employee_writer.writerow(["Pearson"]+ [all_pearson_r[rn] for rn in REWARD_NAME])
    employee_writer.writerow(["Spearman"]+ [all_spearman_r[rn] for rn in REWARD_NAME])

human_eval.to_csv('self-play-est.csv', index=False)