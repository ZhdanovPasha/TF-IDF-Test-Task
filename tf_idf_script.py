import pandas as pd
import collections
import math

    
# Функция для предобработки текста
def transform_text(text):
    good_symbols = ' -'
    # заменим перевод строки на пробел (так можно отделить порядковый номер главы от текста)
    text = text.replace('\n', ' ')
    # оставляем только буквы, пробелы и -
    text = ''.join([symb for symb in text if symb.isalpha() or symb in good_symbols])
    # удаляем слова, которые равные '-'
    # таким образом оставим - внутри слов
    tokens = [token for token in text.split() if token != '-' * len(token)]
    # есть слова, которые начинаются или заканчиваются на '-'
    # не ясно почему, в них удалим первый символ или последний символ соответственно
    tokens = [token[1:] if token.startswith('-') else token for token in tokens]
    tokens = [token[:-1] if token.endswith('-') else token for token in tokens]
    text = ' '.join(tokens)
    # приводим текст к нижнему регистру
    text = text.lower()
    # отделяем порядковый номер главы
    tokens = text.split()[1:]
    
    return ' '.join(tokens)

def tf(text):
    tokens = text.split()
    tf_dict = dict(collections.Counter(tokens))
    for key in tf_dict.keys():
        tf_dict[key] = tf_dict[key] / len(tokens)
    return tf_dict
    
def idf(documents, words):
    df_dict = {
        word: 0
        for word in words
    }
    for document in documents:
        tokens = set(document.split())
        for token in tokens:
            df_dict[token] = df_dict[token] + 1
    for word in words:
        df_dict[word] = math.log(len(documents) / df_dict[word])
    
    return df_dict
        
def get_all_words_from_text(texts):
    words = set()
    for text in texts:
        for token in text.split():
            words.add(token)
            
    return sorted(words)

def tf_idf(documents):
    words = get_all_words_from_text(documents)
    idf_dict = idf(documents, words)
    tf_idf_table = [
        [0 for _ in range(len(documents))]
        for word in words
    ]

    df = pd.DataFrame(tf_idf_table)
    df.index = words
    df.rename({i: 'Песнь {}'.format(i + 1) for i in range(len(df.columns))}, axis=1, inplace=True)
    for i, document in enumerate(documents):
        tf_idf_dict = tf(document)
        for word in tf_idf_dict.keys():
            tf_idf_dict[word] = tf_idf_dict[word] * idf_dict[word]
        
        df[df.columns[i]] = df.index.to_series().map(tf_idf_dict)
            
    df.fillna(0, inplace=True)
    
    return df


with open('gomer01.txt', 'r', encoding='utf-8') as f:
    text = f.read()    
    
# Делим тексты по слову ПЕСНЬ
songs = text.split("ПЕСНЬ ")
# Пропускаем первый текст, поскольку он является введением
songs = songs[1:]
# В 6-ой, 9-ой и 10-ой поэме дополнительно удаляем пункт ПРИМЕЧАНИЕ
chapters_with_additional_info = [5, 8, 9]
for chapter in chapters_with_additional_info:
    songs[chapter] = songs[chapter].split('ПРИМЕЧАНИЯ')[0]

transformed_songs = [transform_text(song) for song in songs]
    
df = tf_idf(transformed_songs)
df.to_csv('gomer_tf_idf.csv', encoding='utf-8')