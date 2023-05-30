import pandas as pd
import os
import simple_icd_10 as icd
from tqdm import tqdm

os.chdir('/home/develop/workspace/Insurance')


letter_dict = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M',
               14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'}

def has_letters(s):
    for c in s:
        if c.isalpha():
            return True
    return False

if __name__ == '__main__':
    data_claim = pd.read_excel('data/1.理赔数据.xlsx')
    data_icdo = pd.read_csv('data/ICD-O.csv')
    dict_icdo = my_dict = dict(zip(data_icdo['code'], data_icdo['value']))
    ENdes_list = []
    code_list = []
    for reason in tqdm(data_claim['Reason2']):
        code = reason.split(":")[0]
        if code == 'C85.0':
            des = 'Other specified and unspecified types of non-Hodgkin lymphoma'
        elif code == 'N18.8':
            des = 'Other chronic renal failure'
        elif code == 'D76.0':
            des = 'Langerhans cell histiocytosis, not elsewhere classified'
        elif code == 'X64.5':
            des = icd.get_description('X64')
        elif code == 'N18.0':
            des = 'End-stage renal disease'
        elif code == 'X78.0':
            des = icd.get_description('X78')
        elif code[-1] == '+':
            des = icd.get_description(code[:-1])
        else:
            try:
                des = icd.get_description(code)
            except:
                if code[0] == 'M':
                    des = dict_icdo[code]
                elif code[0] in ['W', 'X'] and code[-2] == '.':
                    des = icd.get_description(code[:-2])
                elif code[-2] == '.':
                    des = icd.get_description(code[:-2])
                elif not has_letters(code):
                    code_letter = letter_dict[int(code[:2])]
                    first_part = code[2:-1]
                    last_part = code[-1]
                    if last_part == '0':
                        code = f"{code_letter}{first_part}"
                    else:
                        code = f"{code_letter}{first_part}.{last_part}"
                    des = icd.get_description(code)
                else:
                    raise RuntimeError(code)

        ENdes_list.append(des)
        code_list.append(code)
    data_claim['Reason2_CN'] = data_claim['Reason2'].copy()
    data_claim['Reason2'] = ENdes_list
    data_claim['Reason2-Code'] = code_list
    data_claim.to_csv("results/claim_ENreason.csv", encoding='utf_8_sig', index=None)
