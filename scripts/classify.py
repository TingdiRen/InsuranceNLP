import pandas as pd
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import simple_icd_10 as icd
from transformers import pipeline


class Classify_Dataset(Dataset):
    def __init__(self, reason1, reason2, premise, state, prior_f=None):
        if prior_f is not None:
            reason2 = prior_f(reason2)
        reason2 = [i.split(":")[-1] for i in reason2]
        concat_reason = [f"death by {i.lower()} and {j.lower()}" for i, j in zip(reason1, reason2)]
        # condition = lambda x: x == 'entailment'
        # pres = [f'This example is {i}.' for i in premise]
        # hyps = list(reason1)

        self.data = concat_reason
        self.pres1 = list(premise)
        self.i = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    # def __getitem__(self, idx):
    #     h = self.data[idx]
    #     p = self.pres1
    #     return h, p
    def __next__(self):
        if self.i >= self.__len__():
            self.i = 0
            raise StopIteration

        h = self.data[self.i]
        p = self.pres1
        self.i += 1
        return h, p


def prior_f(data):
    new_data = []
    data_icdo = pd.read_csv('data/ICD-O.csv')
    dict_icdo = dict(zip(data_icdo['code'], data_icdo['value_cn']))
    '''分割code和des'''
    for i in data:
        code, desc = i.split(":")
        if code[0] == 'M' and (not icd.is_valid_item(code)):  # ICD-O的代码是错误的描述
            desc = dict_icdo[code]
        new_data.append(desc)
    new_data = [i.split("，")[0] for i in new_data]

    medical_prior = {'蛛网膜': '脑和蛛网膜',
                     '其他': 'other'}
    final_data = []
    for i in new_data:
        tmp = [j for j in medical_prior.keys() if j in i]
        if len(tmp) > 0:
            new_i = i.replace(tmp[0], medical_prior[tmp[0]])
        else:
            new_i = i
        final_data.append(new_i)
    return final_data


def infer_classify(data, pred_cause_level):
    namespaces = locals()

    # for level in range(1, 4):
    #     '''X级死因: list'''
    #     namespaces[f'premise_level{level}'] = [f'This example is {i}.' for i in data[f"list_ruled_death_cause_level{level}"]]

    '''init pretrained model'''
    classifier1 = pipeline("zero-shot-classification", model='./pretrained_cache/models--roberta_large',
                           device=data['args'].gpu_id, batch_size=data['args'].batch_size, multi_label=True)
    # nli_model = pipeline("zero-shot-classification", model=data['args'].pretrained_model, batch_size=8)
    classifier2 = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                           device=data['args'].gpu_id, batch_size=data['args'].batch_size, multi_label=True)

    '''init dataset'''
    dataloader = Classify_Dataset(reason1=data['data_claim_death_cause_reason1'].to_list(),
                                  reason2=data['data_claim_death_cause_reason2'].to_list(),
                                  premise=data[f"list_ruled_death_cause_level{pred_cause_level}"],
                                  state=data['data_claim_state'].to_list(),
                                  prior_f=prior_f)

    '''infer'''
    res = []
    data_claim_reason_classify = data['data_claim'].copy()
    # for level in range(1, 4):
    # premise_lists = namespaces[f'premise_level{level}']
    # namespaces[f'prob_causelevel{level}'] = []
    # namespaces[f'result_causelevel{level}'] = []
    prob1_cause, result1_cause = [], []
    prob2_cause, result2_cause = [], []
    for h, ps in tqdm(dataloader, desc=f'[TASK] Classify [LEVEL] {pred_cause_level}'):
        pred1 = classifier1(h, ps)
        pred2 = classifier2(h, ps)
        # contradiction, neutral, entailment = pred1.logits[0]
        # prob.append(entailment.item())
        # prob = torch.nn.functional.softmax(torch.tensor(prob), dim=0)
        prob1_cause.append(pred1['scores'])
        result1_cause.append(pred1['labels'][np.argmax(pred1['scores'])])
        prob2_cause.append(pred2['scores'])
        result2_cause.append(pred2['labels'][0])
    data_claim_reason_classify[f'prob1_causelevel{pred_cause_level}'] = prob1_cause
    data_claim_reason_classify[f'result1_causelevel{pred_cause_level}'] = result1_cause
    data_claim_reason_classify[f'prob2_causelevel{pred_cause_level}'] = prob2_cause
    data_claim_reason_classify[f'result2_causelevel{pred_cause_level}'] = result2_cause
    data_claim_reason_classify.to_csv(
        os.path.join(data['args'].save_dir, f'Classify_chunk_causelevel{pred_cause_level}_{data["args"].gpu_id}.csv'),
        encoding='utf_8_sig', index=None)
    return data_claim_reason_classify


def concat_df(args, cause_level, chunk_head='Classify'):
    raw_data = pd.read_csv(os.path.join(args.data_dir, 'NLI_final_res.csv'))
    raw_data = raw_data.drop(list(raw_data.columns)[1:], axis=1)
    concat_list = []
    for chunk in range(args.chunk_size):
        concat_list.append(
            pd.read_csv(os.path.join(args.save_dir, f'{chunk_head}_chunk_causelevel{cause_level}_{chunk}.csv')))
    concated_df = pd.concat(concat_list, axis=0)
    sorted_concated_df = pd.merge(raw_data, concated_df, on='ID')
    sorted_concated_df.to_csv(os.path.join(args.save_dir, f"{chunk_head}_causelevel{cause_level}_final_res.csv"),
                              encoding='utf_8_sig',
                              index=None)
    return sorted_concated_df


def read_data(args):
    namespaces = locals()
    '''理赔数据: df'''
    # namespaces['data_claim'] = pd.read_csv(os.path.join(args.data_dir, 'NLI_final_res.csv'))
    namespaces['data_claim'] = pd.read_csv(os.path.join(args.data_dir, 'unsuper1.csv'))
    # namespaces['data_claim'] = namespaces['data_claim'][namespaces['data_claim']['state'] == 'entailment']
    # namespaces['data_claim'] = namespaces['data_claim'][namespaces['data_claim'].isnull().any(axis=1)]
    # namespaces['data_claim'] = namespaces['data_claim'][namespaces['data_claim']['reason_level2'] == '肿瘤']
    if args.chunk_size > 0:
        namespaces['data_claim'] = namespaces['data_claim'][args.gpu_id::args.chunk_size]

    namespaces['data_claim_state'] = namespaces['data_claim']['state']
    for level in range(1, 3):
        '''Reason1.2, Series'''
        namespaces[f"data_claim_death_cause_reason{level}"] = namespaces['data_claim'][f'Reason{level}_EN']

    '''死因: df'''
    namespaces['data_ruled_death_cause'] = pd.read_excel(os.path.join(args.data_dir, '字段解释.xlsx'), sheet_name=2)
    for level in range(1, 4):
        '''X级死因: list'''
        namespaces[f"list_ruled_death_cause_level{level}"] = namespaces[
                                                                 'data_ruled_death_cause'].iloc[:,
                                                             level - 1].dropna().to_list()

    namespaces[f"list_ruled_death_cause_level2"] = pd.Series(
        ["death declared by court's judgment",
         "death by homicide",
         "death by traffic accidents",
         "death by unknown accidents without causes",
         "diseases of tumour",
         "diseases of circulatory system and blood system",
         "diseases of respiratory system",
         "diseases of digestive system", "diseases of urinary and reproductive systems",
         "diseases of nervous or motor system",
         "diseases of endocrine or immune or metabolic system",
         "simply diseases",
         "unknown death without causes",
         "natural death",
         "death or injury by falling",
         "death or injury by drowning",
         "death or injury by suffocation",
         "death or injury by fire",
         "death or injury by scald",
         "death or injury by electric currents, radiation, and extreme environments",
         "death or injury by natural disaster",
         "death or injury by poisonous plant and animal",
         "death or injury by toxic chemicals",
         "death or injury by excessive physical exertion, starvation, thirst",
         "died during surgery"])
    namespaces[f"list_ruled_death_cause_level1"] = pd.Series(
        ["death by disease", "death by accident", "deliberate suicide", "death declared by court's judgment",
         "natural death"])
    return namespaces


def load_args():
    os.chdir('/home/develop/workspace/Insurance')
    parser = argparse.ArgumentParser(description='Death Causes Classification')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=16, help='size for data_claim batch')
    parser.add_argument('--chunk_size', type=int, default=4, help='size for data_claim chunking')
    parser.add_argument('--data_dir', type=str, default='./data', help='directory of data_claim')
    parser.add_argument('--save_dir', type=str, default='./results', help='directory of saving data_claim')
    parser.add_argument('--pretrained_model', type=str, default='pretrained_cache/models--roberta_large',
                        help='directory of pretrained model')
    parser.add_argument('--pred_cause_level', type=int, default=2, help='directory of pretrained model')
    args = parser.parse_args()
    args.device = f'cuda:{args.gpu_id}' if args.gpu_id >= 0 else 'cpu'
    if args.chunk_size > 0 and args.gpu_id < 0:
        raise RuntimeError('chunk_size is not supported for cpu')
    return args


if __name__ == '__main__':
    args = load_args()
    data_claim = read_data(args)
    data_claim_reason_classify = infer_classify(data_claim, args.pred_cause_level)
    # concat_df(args, cause_level=args.pred_cause_level, chunk_head='Classify')
