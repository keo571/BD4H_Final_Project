import os
import re
import json
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize


def find_concept(first_part):
    return re.search(r'"(.*?)"', first_part).group(1)


def find_location(first_part):
    loc = first_part.split()[-2:]
    # First line of the report starts as line number 1, so we need to subtract 1
    start_line_indx = int(loc[0].split(':')[0]) - 1
    start_word_indx = int(loc[0].split(':')[1])
    end_word_indx = int(loc[1].split(':')[1])
    return start_line_indx, start_word_indx, end_word_indx


def find_assertion(third_part):
    return re.search(r'"(.*?)"', third_part).group(1)


def mark_sentence(txt, start_line_indx, start_word_indx, end_word_indx):
    sentence = txt[start_line_indx].split()
    sentence.insert(start_word_indx, '<c>')
    sentence.insert(end_word_indx + 2, '</c>')
    sentence = ' '.join(sentence)
    return sentence


def tokenize(sentence):
    sentence = word_tokenize(sentence)
    sentence = ' '.join(sentence)
    sentence = sentence.replace('< c >', '<c>')
    sentence = sentence.replace('< /c >', '</c>')
    sentence = sentence.split()
    return sentence


def convert(line, txt, id):
    line_lst = line.split('||')
    concept = find_concept(line_lst[0])
    start_line_indx, start_word_indx, end_word_indx = find_location(
        line_lst[0])
    sentence = mark_sentence(txt, start_line_indx,
                             start_word_indx, end_word_indx)
    sentence = tokenize(sentence)
    assertion = find_assertion(line_lst[2])
    meta = dict(
        id=id,
        concept=concept,
        sentence=sentence,
        assertion=assertion
    )
    return meta


def preprocess(ast_folder_path, txt_folder_path, des_path, id):
    with open(des_path, 'w', encoding='utf-8') as fw:
        for filename in os.listdir(ast_folder_path):
            with open(os.path.join(ast_folder_path, filename), 'r', encoding='utf-8') as fr:
                ast = fr.readlines()
            filename = filename.replace('ast', 'txt')
            with open(os.path.join(txt_folder_path, filename), 'r', encoding='utf-8') as fr:
                txt = fr.readlines()
            for line in ast:
                meta = convert(line, txt, ID)
                json.dump(meta, fw, ensure_ascii=False)
                fw.write('\n')
                id += 1
    return id


def preprocess_negex(src_path, des_path, starting_id):
    df = pd.read_csv(src_path, sep='\t')
    df = df.rename(columns={'Condition': 'concept',
                   'negation_status (negated, affirmed, possible)': 'assertion'})
    df = df[['concept', 'sentence', 'assertion']]
    df['assertion'] = np.where(
        df['assertion'] == 'Affirmed', 'present', 'absent')
    with open(des_path, 'w', encoding='utf-8') as fw:
        for index, row in df.iterrows():
            concept = row['concept']
            sentence = row['sentence']
            if concept.lower() not in sentence.lower():
                temp = sentence
                sentence = concept
                concept = temp

            sentence = sentence.replace(concept, ' <c> '+concept+' </c> ')
            sentence = tokenize(sentence)
            meta = dict(
                id=index + starting_id + 1,
                concept=concept,
                sentence=sentence,
                assertion=row['assertion']
            )
            json.dump(meta, fw, ensure_ascii=False)
            fw.write('\n')


if __name__ == '__main__':
    BID_PH_src_path = 'data/original/BID_PH/'
    BID_PH_des_path = 'data/BID_PH.json'
    UPMC_src_path = 'data/original/UPMC/'
    UPMC_des_path = 'data/UPMC.json'
    NegEx_src_path = 'data/original/NegEx_Corp/rsAnnotations-1-120-random.txt'
    NegEx_des_path = 'data/NegEx.json'

    ID = 1

    # BID/PH
    ast_folder_path = BID_PH_src_path + 'ast'
    txt_folder_path = BID_PH_src_path + 'txt'
    ID = preprocess(ast_folder_path, txt_folder_path, BID_PH_des_path, ID)

    # UMPC
    ast_folder_path = UPMC_src_path + 'ast'
    txt_folder_path = UPMC_src_path + 'txt'
    ID = preprocess(ast_folder_path, txt_folder_path, UPMC_des_path, ID)

    # NegEx
    preprocess_negex(NegEx_src_path, NegEx_des_path, ID)
