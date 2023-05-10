import pandas as pd
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import pipeline
from collections import OrderedDict
import torch.nn.utils.rnn as rnn_utils
from transformers import ZeroShotClassificationPipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import simple_icd_10 as icd


def load_args():
    os.chdir('/home/develop/workspace/Insurance')
    parser = argparse.ArgumentParser(description='Death Causes Classification')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=4, help='size for data_claim batch')
    parser.add_argument('--chunk_size', type=int, default=4, help='size for data_claim chunking')
    parser.add_argument('--data_dir', type=str, default='./data', help='directory of data_claim')
    parser.add_argument('--save_dir', type=str, default='./results', help='directory of saving data_claim')
    parser.add_argument('--pretrained_model', type=str, default='pretrained_cache/models--roberta_large',
                        help='directory of pretrained model')
    args = parser.parse_args()
    args.device = f'cuda:{args.gpu_id}' if args.gpu_id >= 0 else 'cpu'
    if args.chunk_size > 0 and args.gpu_id < 0:
        raise RuntimeError('chunk_size is not supported for cpu')
    return args


class NLI_Dataset(Dataset):
    def __init__(self, premise, hypothesis, prior_f=None):
        if prior_f is not None:
            priored_premise = prior_f(premise)
        self.premise = [f'This example is {i}.' for i in priored_premise]
        self.hypothesis = hypothesis

    def __len__(self):
        return len(self.premise)

    def __iter__(self):
        for p, h in zip(self.premise, self.hypothesis):
            yield p, h


def prior_f(data):
    new_data = []
    data_icdo = pd.read_csv('data/ICD-O.csv')
    dict_icdo = dict(zip(data_icdo['code'], data_icdo['value']))
    '''分割code和des'''
    for i in data:
        code, desc = i.split(":")
        if code[0] == 'M' and (not icd.is_valid_item(code)): # ICD-O的代码是错误的描述
            desc = dict_icdo[code]
        new_data.append(desc)
    new_data = [i.split("，")[0] for i in new_data]

    medical_prior = {'蛛网膜': '脑和蛛网膜',
                     '贲门': '胃和贲门',
                     '其他': 'other'}
    final_data = []
    for i in tqdm(new_data, desc='[Perparing Data]'):
        tmp = [j for j in medical_prior.keys() if j in i]
        if len(tmp) > 0:
            new_i = i.replace(tmp[0], medical_prior[tmp[0]])
        else:
            new_i = i
        final_data.append(new_i)
    return final_data


def infer_NLI(data):
    '''
    :param data:
    :return:
    '''

    '''init dataset'''
    dataloader = NLI_Dataset(premise=data['data_claim_death_cause_reason2'].to_list(),
                             hypothesis=data['data_claim_death_cause_reason1'].to_list(),
                             prior_f=prior_f)
    '''init pretrained model'''
    tokenizer = AutoTokenizer.from_pretrained(data['args'].pretrained_model)
    nli_model = AutoModelForSequenceClassification.from_pretrained(data['args'].pretrained_model).to(
        data['args'].device)

    res = []
    data_claim_reason_compare = data['data_claim'].copy()
    for p, h in tqdm(dataloader, desc='[TASK] NLI'):
        input = tokenizer.encode(h, p, return_tensors='pt').to(data['args'].device)
        pred = nli_model(input)
        contradiction, neutral, entailment = torch.nn.functional.softmax(pred.logits[0], dim=0)
        res.append((contradiction.item(), neutral.item(), entailment.item()))
    data_claim_reason_compare['contradiction'], data_claim_reason_compare['neutral'], data_claim_reason_compare[
        'entailment'] = zip(*res)
    data_claim_reason_compare['state'] = data_claim_reason_compare.iloc[:, -3:].idxmax(axis=1)
    data_claim_reason_compare['prob'] = data_claim_reason_compare.iloc[:, -4:-1].max(axis=1)
    data_claim_reason_compare.to_csv(os.path.join(data['args'].save_dir, f'NLI_chunk_{data["args"].gpu_id}.csv'),
                                     encoding='utf_8_sig', index=None)
    return data_claim_reason_compare


def infer_Classify(data, reason_level=1, cause_level=2):
    '''
    :param data:
    :param reason_level:
    :param cause_level:
    :return:
    '''

    '''init dataset'''
    dataloader = NLI_Dataset(premise=[i.split(":")[1] for i in data[f'list_ruled_death_cause_level{cause_level}']],
                             hypothesis=data[f'data_claim_death_cause_reason{reason_level}'].to_list())

    '''init pretrained model'''
    tokenizer = AutoTokenizer.from_pretrained(data['args'].pretrained_model)
    nli_model = AutoModelForSequenceClassification.from_pretrained(data['args'].pretrained_model).to(
        data['args'].device)

    '''infer with chunk'''
    res = []
    data_claim_classify = data['data_claim'].copy()
    for p, h in tqdm(dataloader, desc='[TASK] Classify'):
        input = tokenizer.encode(h, p, return_tensors='pt').to(data['args'].device)
        pred = nli_model(input)
        contradiction, neutral, entailment = torch.nn.functional.softmax(pred.logits[0])
        res.append(dict(zip(pred['labels'], pred['scores'])))

    data_claim_classify['pred'] = res
    data_claim_classify.to_csv(os.path.join(data['args'].save_dir, f'Classify_chunk_{data["args"].gpu_id}.csv'),
                               encoding='utf_8_sig', index=None)
    return data_claim_classify


def concat_df(args, chunk_head='NLI'):
    raw_data = pd.read_excel(os.path.join(args.data_dir, '1.理赔数据.xlsx'))
    raw_data = raw_data.drop(list(raw_data.columns)[1:], axis=1)
    concat_list = []
    for chunk in range(args.chunk_size):
        concat_list.append(pd.read_csv(os.path.join(args.save_dir, f'{chunk_head}_chunk_{chunk}.csv')))
    concated_df = pd.concat(concat_list, axis=0)
    sorted_concated_df = pd.merge(raw_data, concated_df, on='ID')
    sorted_concated_df.to_csv(os.path.join(args.save_dir, f"{chunk_head}_final_res.csv"), encoding='utf_8_sig',
                              index=None)
    return sorted_concated_df


def read_data(args):
    namespaces = locals()
    '''理赔数据: df'''
    namespaces['data_claim'] = pd.read_excel(os.path.join(args.data_dir, '1.理赔数据.xlsx'))
    # namespaces['data_claim'] = pd.read_csv(os.path.join(args.data_dir, 'claim_ENreason.csv'))
    if args.chunk_size > 0:
        namespaces['data_claim'] = namespaces['data_claim'][args.gpu_id::args.chunk_size]

    for level in range(1, 3):
        '''Reason1.2, Series'''
        namespaces[f"data_claim_death_cause_reason{level}"] = namespaces['data_claim'][f'Reason{level}']

    '''死因: df'''
    namespaces['data_ruled_death_cause'] = pd.read_excel(os.path.join(args.data_dir, '字段解释.xlsx'), sheet_name=2)
    for level in range(1, 4):
        '''X级死因: list'''
        namespaces[f"list_ruled_death_cause_level{level}"] = namespaces[
                                                                 'data_ruled_death_cause'].iloc[:, level - 1].to_list()

    return namespaces


if __name__ == '__main__':
    args = load_args()
    # data_claim = read_data(args)
    concat_df(args, chunk_head='NLI')
    # data_claim_reason_compare = infer_NLI(data_claim)
    # data_claim_classify = infer_classify(data_claim)
