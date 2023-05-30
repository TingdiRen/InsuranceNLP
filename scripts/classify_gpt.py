import pandas as pd
import os
from tqdm import tqdm
import time
import argparse
import simple_icd_10 as icd
import requests
from requests.exceptions import Timeout, HTTPError

def query(payload, API_URL, headers):
    while True:
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=2)
            if response.status_code == 200:
                break
            else:
                continue
        except Timeout:
            continue
    return response.json()


class Classify_Dataset:
    def __init__(self, reason1, reason2, level, pred_cause_level2=None):
        self.init_template()

        self.level = level
        self.reason1 = reason1
        self.reason2 = self.prior_f(reason2)
        if level == 3:
            self.pred_cause_level2 = pred_cause_level2
        else:
            self.pred_cause_level2 = [None for i in self.reason1]

        self.i = 0

    def __next__(self):
        if self.i >= self.__len__():
            self.i = 0
            raise StopIteration
        
        if self.pred_cause_level2[self.i] in self.level3_pass:
            data = self.pred_cause_level2[self.i]
        else:
            data = f'Please answer to the following question. If someone dead with symptom "{self.reason2[self.i].lower()}", which is caused by event "{self.reason1[self.i].lower()}", what\'s the proximate cause of the death in insurance? ' \
               f'Chose an answer from {self.get_cause(self.i)}.'
        self.i += 1
        return data

    def get_cause(self, index):
        if self.level == 2:
            return self.cause_level2
        elif self.level == 3:
            pred_cause_level2 = self.pred_cause_level2[index]
            return self.cause_level3_dict[pred_cause_level2]

    def init_template(self):
        self.cause_level2 = '["Declared dead by the court", ' \
                            '"Intentional suicide", "Homicide", ' \
                            '"Accidents caused by non-vehicles", "Accidents caused by vehicles", "Unexplained accident", ' \
                            '"Tumor", "Diseases of circulatory and blood system", "Diseases of Respiratory system", ' \
                            '"Diseases of the digestive system", "Diseases of the urinary and reproductive system", ' \
                            '"Diseases of the nervous and motor systems", "Endocrine/immune/metabolic diseases", ' \
                            '"Diseases that cannot be classified as tumors or a specific system", ' \
                            '"Unable to determine the disease that caused the death"]'

        self.level3_pass = ["Declared dead by the court","Intentional suicide","Homicide","Unexplained accident","Diseases that cannot be classified as tumors or a specific system","Unable to determine the disease that caused the death"]

        self.cause_level3_dict = {"Non-Traffic Accident": '["Falling down and falling","Inanimate mechanical force","With life mechanical force","Drowning","Suffocation","Fire","Scalding","Electric current, radiation and extreme environment","Natural disasters","Toxic plants and animals","Toxic chemicals","Death from physical overexertion","Death from accidents during medical treatment"]',
                                  "Traffic Accident": '["Killed by vehicles while walking","Non-motorized vehicles accident","Two-wheeled motor vehicles accident","Three-wheeled motor vehicles accident","Car accident","Passenger car accident","Lorry accident","special vehicle accident","Rail accident","Ship accident","Aircraft accident"]',
                                  "Tumor": '["Malignant Tumors","Benign brain tumors","Hemangioma or aneurysm","Other benign tumors"]',
                                  "Diseases of circulatory and blood system": '["Acute myocardial infarction","Cerebral Stroke","Coronary heart disease","Aortic disease","Heart valve disease","Heart disease of pulmonary origin","Heart Inflammation","Aplastic anemia","Other circulatory and blood disorders"]',
                                  "Diseases of the digestive system": '["Acute Liver Failure","Chronic liver failure","Crohn\'s disease and ulcerative colitis","Pancreatitis","Gallbladder inflammation","Other digestive system diseases"]',
                                  "Diseases of Respiratory system": '["Acute respiratory failure, influenza, acute pneumonia","Chronic respiratory failure, chronic obstructive pulmonary disease, asthma, pulmonary fibrosis progression of respiratory failure"]',
                                  "Diseases of the urinary and reproductive system": '["Acute renal failure, acute renal failure due to inadequate renal blood supply or abnormal renal excretory function","Chronic renal failure, uremia","Obstetric diseases	","Sexually transmitted diseases, excluding AIDS","Other urological and reproductive system diseases"]',
                                  "Diseases of the nervous and motor systems": '["Alzheimer\'s disease and other dementias","Parkinson\'s disease","Epilepsy","Motor neuron disease","Multiple sclerosis","Myasthenia Gravis","Myotonic Dystrophy","Encephalitis and Meningitis","Other neurological and motor system diseases"]',
                                  "Endocrine/immune/metabolic diseases": '["Diabetes","Lupus Erythematosus","Rheumatoid Arthritis","Scleroderma","AIDS","Other endocrine/immune/metabolic diseases"]'}

    def prior_f(self, data):
        new_data = []
        '''split code and des'''
        for i in data:
            code, *desc = i.split(":")
            if code == 'R99':
                desc = 'death'
                new_data.append(desc)
            else:
                new_data.append(''.join(desc))
        return new_data

    def __len__(self):
        return len(self.reason1)

    def __iter__(self):
        return self


def infer_classify(data, pred_cause_level):
    namespaces = locals()

    '''init infer API'''
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
    API_TOKEN = data['args'].API_TOKEN
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    '''init dataset'''
    if pred_cause_level == 2:
        dataloader = Classify_Dataset(reason1=data['data_claim_death_cause_reason1'].to_list(),
                                      reason2=data['data_claim_death_cause_reason2'].to_list(),
                                      level=pred_cause_level)
    elif pred_cause_level == 3:
        dataloader = Classify_Dataset(reason1=data['data_claim_death_cause_reason1'].to_list(),
                                      reason2=data['data_claim_death_cause_reason2'].to_list(),
                                      level=pred_cause_level,
                                      pred_cause_level2=data['data_claim_death_cause_level2']['resultGPT_causelevel2'].to_list())

    '''infer'''
    result_cause = []
    data_claim_reason_classify = data['data_claim'].copy()
    for q in tqdm(dataloader, desc=f'[Chunk] {data["args"].chunk_id} [LEVEL] {pred_cause_level}'):
        if q[:6] == 'Please':
            output = query({"inputs": q}, API_URL, headers)  # request for inference API
            while 'error' in output:
                output = query({"inputs": q}, API_URL, headers)
                time.sleep(10)
            result_cause.append(output[0]['generated_text'])
        else:
            result_cause.append(q)
    data_claim_reason_classify[f'resultGPT_causelevel{pred_cause_level}'] = result_cause
    data_claim_reason_classify.to_csv(
        os.path.join(data['args'].save_dir, f'Classify_chunk_causelevel{pred_cause_level}_{data["args"].chunk_id}.csv'),
        encoding='utf_8_sig', index=None)
    return data_claim_reason_classify


def concat_df(args, cause_level, chunk_head='Classify'):
    raw_data = pd.read_csv(os.path.join(args.data_dir, 'processed_claim.csv'))
    raw_data = raw_data.drop(list(raw_data.columns)[1:], axis=1)
    concat_list = []
    for chunk in range(args.chunk_size):
        concat_list.append(
            pd.read_csv(os.path.join(args.save_dir, f'{chunk_head}_chunk_causelevel{cause_level}_{chunk}.csv')))
    concated_df = pd.concat(concat_list, axis=0)
    sorted_concated_df = pd.merge(raw_data, concated_df, on='ID')
    sorted_concated_df.to_csv(os.path.join(args.save_dir, f"{chunk_head}_causelevel{cause_level}_final_res.csv"),
                              encoding='utf_8_sig', index=None)
    return sorted_concated_df


def read_data(args):
    namespaces = locals()
    namespaces['data_claim'] = pd.read_csv(os.path.join(args.data_dir, 'processed_claim.csv'))[
                               args.chunk_id::args.chunk_size]

    for level in range(1, 3):
        '''Reason1.2, Series'''
        namespaces[f"data_claim_death_cause_reason{level}"] = namespaces['data_claim'][f'Reason{level}_EN']

    if args.pred_cause_level == 3:
        namespaces[f"data_claim_death_cause_level2"] = pd.read_csv(
            os.path.join(args.data_dir, 'Classify_causelevel2_final_res.csv'))[args.chunk_id::args.chunk_size]
    return namespaces


def load_args():
    # os.chdir('../')
    os.chdir('/home/develop/workspace/Insurance')
    parser = argparse.ArgumentParser(description='Death Causes Classification')
    parser.add_argument('--chunk_id', type=int, default=0, help='chunk id')
    parser.add_argument('--batch_size', type=int, default=16, help='size for data_claim batch')
    parser.add_argument('--chunk_size', type=int, default=8, help='size for data_claim chunking')
    parser.add_argument('--data_dir', type=str, default='./data', help='directory of data_claim')
    parser.add_argument('--save_dir', type=str, default='./results', help='directory of saving data_claim')
    parser.add_argument('--pretrained_model', type=str, default='pretrained_cache/models--roberta_large',
                        help='directory of pretrained model')
    parser.add_argument('--API_TOKEN', type=str, default='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',
                        help='api token for hugging face')
    parser.add_argument('--pred_cause_level', type=int, default=2, help='level for pred cause')
    parser.add_argument('--concat', type=bool, default=False, help='concat chunk results')
    args = parser.parse_args()
    if args.chunk_size > 0 and args.chunk_id < 0:
        raise RuntimeError('chunk_size is not supported for cpu')
    return args


if __name__ == '__main__':
    args = load_args()
    if args.concat:
        concat_df(args, cause_level=args.pred_cause_level, chunk_head='Classify')
    else:
        data_claim = read_data(args)
        data_claim_reason_classify = infer_classify(data_claim, args.pred_cause_level)
