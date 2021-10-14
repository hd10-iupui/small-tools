import csv
import os
import time
import numpy as np

from configparser import ConfigParser
from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.extractor import extract_candidates


"""Use Stanford to candidates first (keep the order as the same as within the sentence, also keep repeat candidate), 
then scan the sentence words from beginning to the end, so that give the extracted candidates the index which are their 
word-level position of the sentence.

StanfordNLP and swissom_ai need to be ready before using."""


def load_local_corenlp_pos_tagger():
    """
    cd haoran/IJCAI/HAR_master/ai_research_keyphrase_extraction_master/stanford-corenlp-full-2018-02-27/
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &
    """
    config_parser = ConfigParser()
    config_parser.read('/home/resadmin/haoran/IJCAI/HAR_master/ai_research_keyphrase_extraction_master/config.ini')
    host = config_parser.get('STANFORDCORENLPTAGGER', 'host')
    port = config_parser.get('STANFORDCORENLPTAGGER', 'port')
    return PosTaggingCoreNLP(host, port)


ptagger = load_local_corenlp_pos_tagger()


def str2float(_input_str):
    # print(_input_str)
    _input_str = _input_str[1:-1]
    for _i in range(10):
        if _input_str[-1] == ' ':
            _input_str = _input_str[:-1]
        if _input_str[0] == ' ':
            _input_str = _input_str[1:]

    _input_str = _input_str.replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(', ')
    _out_array = np.array([float(_item) for _item in _input_str])
    return _out_array


start_time = time.time()

path_list = ['Inspec']  # , 'DUC2001'

total_problem_files = []

for p in range(1):

    mid_problems = [path_list[p]]

    project = path_list[p]
    data_path = '/home/resadmin/haoran/GNN/data/'
    doc_path = data_path + 'processed_Inspec/processed_docsutf8/'
    processed_path = data_path + 'processed_' + project + '/'

    files = os.listdir(doc_path)
    for i, file in enumerate(files):
        files[i] = file[:-4]

    files = files[:1]

    print(project, 'docs:', len(files))

    bert_list = ['base']
    for bert_pacakge in bert_list:

        problem_files = [bert_pacakge]

        for n, file in enumerate(files):

                sentence_file = processed_path + 'doc_word_embedding_' + bert_pacakge + '_0816/' + file

                words_avg_embedding_dict = {}
                doc_candidates = []

                with open(sentence_file+'_doc_word_embeddings.csv', newline='') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',')
                    words = []
                    for row in spamreader:
                        k = row[0]
                        v = str2float(row[1])
                        words.append(k)

                # get candidates in this sentence
                #  = ' '.join(words)
                # sentence = sentence.replace(' - ', '-')
                # sentence = sentence.split(' . ')
                sentence = 'recovering lost efficiency of exponentiation algorithms on smart-cards/cats and exponentiation algorithms and algorithms . hello , world !'
                print(sentence)
                tagged = ptagger.pos_tag_raw_text(sentence)

                text_obj = InputTextObj(tagged, 'en')

                candidates, trees = extract_candidates(text_obj, repeat=True)

                print(candidates)

                sentence_words = sentence.split()
                print(sentence_words)

                # we append candidates as their order as also keep the repeat
                remain_index = list(range(len(sentence_words)))
                total_index = remain_index
                for candidate in candidates:
                    candidate_words = candidate.split()
                    local_window = len(candidate_words)

                    print('\ncandidate:',candidate, '   window:', local_window)

                    index_collect = []

                    for d, index in enumerate(remain_index):
                        print(d, 'checking index: ', index, '       remain_index-1', remain_index)  # content

                        if sentence_words[index] == candidate_words[0]:  # content

                            success = 1
                            index_collect.append(index)  # content

                            for ll in range(local_window-1):
                                if sentence_words[index + ll + 1] == candidate_words[ll + 1]:  # content
                                    success+=1
                                    index_collect.append(index + ll + 1)  # content

                            if success == local_window:
                                remain_index = remain_index[local_window:]  # l itself occupy a digit index  # order
                                print(d, '  success index: ', index_collect, 'remain_index-3', remain_index)
                                print('candidate index', index_collect)  # if success, stop check rest index
                                break
                            else:
                                remain_index = remain_index[success:]  # order
                                print(d, 'checking index: ', index, 'remain_index-4', remain_index)

                        else:
                            remain_index = remain_index[1:]  # order  # recursively drop the first one
                            print(d, '  remove index: ', index, '       remain_index-2', remain_index)
                            continue

                doc_candidates += candidates   # merge sentence candidate to doc candidate set
