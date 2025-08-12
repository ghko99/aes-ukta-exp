import torch
from kobert_transformers import get_kobert_model, get_tokenizer
import pandas as pd
from tqdm import tqdm
import csv
import os
import numpy as np
from config import config

def str_to_score_list(str_list):
    score_list = [float(st) for st in str_list.split('#')]
    return score_list

def get_essay_dataset_11_rubrics(args, is_rubric=False):
    df = pd.read_csv(args['dataset_path'], encoding='utf-8-sig')
    essays = [essay.split(args['sentence_sep']) for essay in df[args['essay_key']].to_list()]
    essays = [[sent.strip() for sent in essay if len(sent.strip()) > 0] for essay in essays]
    if config['is_topic_label']:
        for i in range(len(essays)):
            essay = essays[i]
            labeled_essay = [df.iloc[i][args['prompt_key']]] + essay
            essays[i] = labeled_essay
    if is_rubric:
        scores = df['essay_score_avg'].to_list()
        scores = np.array([str_to_score_list(score) for score in scores])
        return essays, scores / args['num_range']
    else: 
        return essays

def embedding(model,tokenizer,essays,args):
    if not os.path.exists(args['emb_file_path']):
        os.makedirs(args['emb_file_path'])
    
    ff = open(os.path.join(args['emb_file_path'], f"{args['dataset_path'].split('/')[1]}_{'notlabeled' if config['is_topic_label'] == False else 'labeled'}.csv"), 'w', newline='')
    writer_ff = csv.writer(ff)

    with torch.no_grad():
        model.eval()
        for i in tqdm(range(len(essays))):
            inputs = tokenizer.batch_encode_plus(essays[i], max_length=args['max_length'], padding='max_length', truncation=True)
            input_ids = torch.tensor(inputs['input_ids']).cuda()
            attention_mask = torch.tensor(inputs['attention_mask']).cuda()
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            embedded_features = out[0].detach().cpu()[:, 0, :].numpy()
            for i in embedded_features:
                writer_ff.writerow(i)
            torch.cuda.empty_cache()
    ff.close()
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_kobert_model().to(device)
    tokenizer = get_tokenizer()

    args = config["aihub_v1"]  # Change to "aihub_v1" for the other dataset
    essays = get_essay_dataset_11_rubrics(args)
    print(essays[0])
    embedding(model,tokenizer,essays,args)
