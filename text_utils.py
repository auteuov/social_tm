from collections import Counter

from gensim.corpora import Dictionary
from nltk import WordNetLemmatizer
from nltk.util import ngrams
import regex
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import pymorphy2
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import string
import os
from stopwords import STOPWORDS

en_lemm = WordNetLemmatizer()
ru_lemm = pymorphy2.MorphAnalyzer()
en_alphabet = string.ascii_letters
ru_alphabet = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"


def lemm(word):
    if any(c in ru_alphabet for c in word):  # if any character in word in en alphabet
        return ru_lemm.parse(word)[0].normal_form
    elif any(c in en_alphabet for c in word):
        return en_lemm.lemmatize(word)
    return word


def clean_word(word, need_lemm=False):
    """
    work with one word
    :param word:
    :return:
    """
    if need_lemm:
        word = lemm(word)
    return word


def text2ngram(texts, n):
    return ngrams(texts, n)


def filter_numeric(word):
    return word != '\n' and word != "\r" \
           and word not in STOPWORDS and len(word) > 2 and not regex.match(r".*\d+.*", word) and not word.startswith(
        "www_") \
           and not word.startswith("http_") and not word.startswith("https_") and not word.endswith("_ru")


def process_line_ngram(cur_id, group_text, ngram, ngram_threshold_count=10, need_lemm=False):
    # all_words = [clean_word(w) for w in group_text.split() if w not in stop_list]
    all_words = [clean_word(w, need_lemm=need_lemm) for w in word_tokenize(group_text) if filter_numeric(w)]
    all_words = [w for w in all_words if w]  # remove empty words

    all_words_ngram = ['_'.join(row) for row in ngrams(all_words, ngram)]
    if ngram == 1:
        assert len(all_words_ngram) == len(all_words)

    words_dict_ngram = Counter(all_words_ngram)
    words_sorted = sorted(words_dict_ngram.items(), key=lambda x: x[1], reverse=True)
    words_text = "{} |text ".format(cur_id) \
                 + ' '.join(['{}:{}'.format(k, v) for k, v in words_sorted])
    if len(words_sorted) == 0:
        words_text = "{} |text empty_line".format(cur_id)
    return words_text


def get_line_vw(i, line, ngram=1, use_tab=True, use_lower=True, use_clear=True, need_lemm=True):
    cur_id, line = get_id_text(i, line, use_tab=use_tab, use_lower=use_lower, use_clear=use_clear)
    if line and cur_id is not None:
        cur_line = process_line_ngram(cur_id, line, ngram, need_lemm=need_lemm)
    else:
        cur_line = "{} |text empty_line".format(cur_id)
    return cur_line


def convert2uci(ifile):
    lines = open(ifile).readlines()
    tokens = {}
    texts = []
    for line in lines:
        items = line.split()
        texts.append(items)
        tokens |= set(items)
    tokens = sorted(tokens)
    for t in tokens:
        pass

    return tokens


def clear_text_characters(text, need_lower=True):
    # use regex, \p Language unicode, pos tagging
    text = text.replace("<br>", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    # text = regex.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=regex.MULTILINE)
    text = regex.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = regex.sub(r'vk.com\/.*\b', '', text)
    text = regex.sub("[^\s\p{L}\p{N}]", "", text)

    if need_lower:
        text = text.lower()
    return text


def get_gensim_text(X_text):
    gensim_texts = [filter_stop_list_line_items(clear_text_characters(line).split()) for line in X_text]
    return gensim_texts


def create_vw(ifile_path, ofile_path_template, use_tab=True, use_lower=True,
              use_clear=True, need_lemm=True, skip_empty_line=True):
    for ngram in [1, 2, 3]:  # ngram
        # print(len(open(ifile_path).readlines()))
        # ofile_path = r"C:\HOME\datasets\vw\vkposts_words_{}_lower.vw".format(n)
        ofile_path = ofile_path_template.format(ngram)
        if not os.path.exists(ofile_path):
            with open(ifile_path, encoding='utf-8') as ifile, \
                    open(ofile_path, 'w', encoding='utf-8') as ofile:
                empty_line_count = 0
                for i, line in enumerate(ifile):
                    if line.startswith("owner_id") or line.startswith("id"):
                        print("skip header")
                        continue
                    line = clear_text_characters(line)
                    vw_line = get_line_vw(i, line, ngram=ngram, use_tab=use_tab, use_lower=use_lower,
                                          use_clear=use_clear, need_lemm=need_lemm)
                    if skip_empty_line and vw_line.endswith("|text empty_line"):
                        print("skip empty line", i)
                        empty_line_count += 1
                        continue
                    else:
                        ofile.write(vw_line + "\n")
            print(ifile)
            print(ofile_path)
            print("total lines", i)
            print("skip lines", skip_empty_line)
        else:
            print("file already exist: ", ofile_path)


def get_texts_from_file(text_file_path, use_tab=False):
    # texts = [["a", "b"], ["cc", "dddd", "eeee"], ...]
    try:
        texts = [clear_text_characters(line).split() for line in open(text_file_path, encoding='utf-8').readlines()]
    except UnicodeDecodeError as ex:
        texts = [clear_text_characters(line).split() for line in open(text_file_path, encoding='cp1251').readlines()]
        print("cp1251 found")

    if use_tab:
        texts = [line[1:] for line in texts]  # drop first id token
    return texts


def get_id_text(line_ind, line, use_tab=True, use_lower=True, use_clear=True):
    if use_tab:
        ind = line.find('\t')
        if ind == -1:
            ind = line.find(" ")

        if ind != -1:
            cur_id = line[:ind]
            text = line[ind + 1:]
        else:
            print("line without tab/space cur_id", line_ind)
            cur_id = line_ind
            text = line
    else:
        cur_id = line_ind
        text = line
    if use_lower:
        text = text.lower()

    if use_clear:
        text = clear_text_characters(text)
    return cur_id, text


def get_docs_raw(ifile_path, use_tab=True, use_lower=True, use_clear=True, encoding='utf-8'):
    cur_ids = []
    texts = []
    with open(ifile_path, encoding=encoding) as ifile:
        for line_ind, line in enumerate(ifile):
            cur_id, text = get_id_text(line_ind, line, use_tab=True, use_lower=True, use_clear=True)
            cur_ids.append(cur_id)
            texts.append(text)
    return cur_ids, texts


def dict2vwline(cur_id, cur_dict):
    cur_dict = {clean_word(k): v for k, v in cur_dict.items() if k and k not in STOPWORDS}  # filter words
    words_sorted = sorted(cur_dict.items(), key=lambda x: x[1], reverse=True)
    words_text = "{} |text ".format(cur_id) \
                 + ' '.join(['{}:{}'.format(k, v) for k, v in words_sorted])
    if len(words_sorted) == 0:
        words_text = "{} |text empty_line".format(cur_id)
    return words_text


def create_vw_tfidf(ifile_path, ofile_path):
    cur_ids, docs_raw = get_docs_raw(ifile_path, use_tab=True)
    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit_transform(docs_raw)
    terms = vectorizer.get_feature_names()
    print(tf_idf.shape)
    del docs_raw
    df_tfidf = pd.DataFrame(tf_idf.toarray(), columns=terms, index=cur_ids)

    with open(ofile_path, 'w', encoding='utf-8') as ofile:
        for cur_id, row in df_tfidf.iterrows():
            cur_dict = {}
            for k, v in row.items():
                if v != 0:
                    cur_dict[k] = v  # save terms with value
            line = dict2vwline(cur_id, cur_dict)
            ofile.write(line + "\n")


def characters_filter_test():
    ifile_path = r"C:\HOME\easy_parse\facebook_first_flight_cities_line_filtered.txt"
    ofile_path = r"C:\HOME\easy_parse\facebook_first_flight_cities_line_filtered_chars.txt"

    with open(ifile_path, encoding='utf-8') as ifile, open(ofile_path, 'w', encoding='utf-8') as ofile:
        for line in ifile:
            ofile.write(clear_text_characters(line).lower() + " ")


def get_hashtags(text):
    hashtags_items = sorted(set(part[1:] for part in text.split() if part.startswith('#')))
    hashtag_line = clear_text_characters(" ".join(hashtags_items))
    return hashtag_line


def get_urls(text):
    # url_pattern = "https?:\/\/\S+\b|www\.(\w+\.)+\S*"
    url_pattern = r'http[s]*?:\/\/(?:[-\w.]|(?:%[\da-fA-F]{2}))+[\/A-Za-z\d]*\b'
    urls_list = regex.findall(url_pattern, text)
    return " ".join(urls_list)


def get_all_modules():
    allmodules = [sys.modules[name] for name in set(sys.modules)]
    return allmodules


def create_urls_file(ifile_path, ofile_path):
    with open(ifile_path, encoding='utf-8') as ifile, \
            open(ofile_path, 'w', encoding='utf-8') as ofile:
        for i, line in enumerate(ifile):
            cur_id, text = get_id_text(i, line, use_clear=False)
            url_line = "{}\t{}".format(cur_id, get_urls(text))
            vw_line = get_line_vw(i, url_line.lower(), use_clear=False)
            ofile.write(vw_line + "\n")
    print(ofile_path)


def create_hashtag_file(ifile_path, ofile_path):
    with open(ifile_path, encoding='utf-8') as ifile, \
            open(ofile_path, 'w', encoding='utf-8') as ofile:
        for i, line in enumerate(ifile):
            cur_id, text = get_id_text(i, line, use_clear=False)
            hash_tag_line = "{}\t{}".format(cur_id, get_hashtags(text))
            vw_line = get_line_vw(i, hash_tag_line.lower(), use_clear=False)
            ofile.write(vw_line + "\n")

    print(ofile_path)


def create_modality_vw(modality_dict, ofile_path):
    for modality, modality_file_path in modality_dict.items():
        print(modality, modality_file_path)


def filter_stop_list(texts, freq_theshold=2):
    return [[w for w in doc if w not in STOPWORDS and len(w) > freq_theshold] for doc in texts]


def filter_stop_list_line_items(line_items, freq_theshold=2):
    return [w for w in line_items if w not in STOPWORDS and len(w) > freq_theshold]


def get_texts_objs(text_file_path, need_filter=False):
    texts = get_texts_from_file(text_file_path)
    if need_filter:
        texts = filter_stop_list(texts)

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return texts, dictionary, corpus


def json2csv(ifile_path, ofile_path, id_col="owner_id", text_col="text"):
    owner_id_list = []
    data = {}
    with open(ifile_path, encoding='utf-8') as ifile:
        for i, line in enumerate(ifile):
            try:
                json_line = json.loads(line)
                cur_id = json_line[id_col]
                line_csv = clear_text_characters(json_line[text_col])
                if line_csv:
                    if cur_id not in data:
                        data[cur_id] = line_csv
                        owner_id_list.append(cur_id)
                    else:
                        data[cur_id] = data[cur_id] + " " + line_csv
            except Exception as ex:
                print("error", i)
                pass
    # 8231
    df = pd.DataFrame({id_col: list(data.keys()), text_col: list(data.values())})
    print("total owners:", len(owner_id_list))
    df.to_csv(ofile_path + "_final.csv", encoding='utf-8', sep='\t')
    print(ofile_path)
