# Run with Command Line
import sys

# Emoji, Symbol, Number, and Englist Letter Removing
import demoji
import re

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# DataFrame
import pandas as pd

# Dump Machine Learning
import pickle

# Word Tokinzing
from pythainlp import Tokenizer
from pythainlp.util import Trie
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus.common import thai_words

stopwords = list(thai_stopwords())

def tokenizing(text):
    # Add Some Unknown Words
    custom_words_list = set(thai_words())
    words = ["พร้อมเพย์", "มายโม"]
    custom_words_list.update(words)

    trie = Trie(custom_words_list)
    tokenizer_newmm = Tokenizer(custom_dict=trie, engine='newmm', keep_whitespace=False)
    
    stopwords_csv = pd.read_csv('./ml/stopwords.csv')
    # stopwords_list = list(stopwords_csv['Word'])
    stopwords_list = ["ทำ", "ไหม", "ผม", "คน", "เรื่อง",  "ตอนนี้", "ทราบ", "น", "บอ", "แบบนี้", "โดน", "ดี", "ขอบคุณ", "รู้", "นะคะ", "บอ",  "กรุงไทย", "กี่", "ชื่อ", "ไทย", "หรอ", "ตอน", "พี่", "ดู", "เค้า", "แม่", "หนู", "เพื่อน", "รบกวน", "สวัสดี", "เจอ", "ที่จะ", "หัวข้อ", "ส", "ค", "พ่อ", "หา", "เหมือน", "กสิกร", "อยากรู้", "ตัว", "งง", 
    "กลับมา", "นึง", "ผู้รู้", "ตัวเอง", "ทำได้", "เก่า", "ท", "ไม่ต้อง", "สำหรับ", "แล้วก็", "อะ", "เข้ามา", "แทน", "หน้า", "ดิฉัน", "ก", "ย", "แล้", "แฟน", "สัก", "ง", "เรียบร้อย", "คับ", "สรุป", "กระทู้", "ธอส", "ออกมา", "ป่าว", "เด็ก", "สงสัย", "เหมือนกัน", "อ่าน", "ต่อไป", "กลัว"]
    new_stopwords = stopwords +  stopwords_list
    
    text = re.sub('เเ','แ', text)
    text = re.sub('เเ','แ', text)
    text = re.sub('แอป','แอพ', text)
    text = re.sub('แอฟ','แอพ', text)
    text = re.sub('แอ๊ปฯ','แอพ', text)
    text = re.sub('แอ๊ป','แอพ', text)
    text = re.sub('แอ็ป','แอพ', text)
    text = re.sub('แอ็พ','แอพ', text)
    text = re.sub('แอพพลิเคชัน','แอพพลิเคชั่น', text)
    text = re.sub('แอพพลิเคชั่น','แอพ', text)
    text = re.sub('application','แอพ', text)
    text = re.sub('Application','แอพ', text)
    text = re.sub('pre approved','อนุมัติ', text)
    text = re.sub('pre approve','อนุมัติ', text)
    text = re.sub('app','แอพ', text)
    text = re.sub('App','แอพ', text)
    text = re.sub('ATM','เอทีเอ็ม', text)
    text = re.sub('atm','เอทีเอ็ม', text)
    text = re.sub('Atm','เอทีเอ็ม', text)
    text = re.sub('MyMo','มายโม', text)
    text = re.sub('mymo','มายโม', text)
    text = re.sub('my mo','มายโม', text)
    text = re.sub('Mymo','มายโม', text)
    text = re.sub('My mo','มายโม', text)
    text = re.sub('My Mo','มายโม', text)
    text = re.sub('ธ\.','ธนาคาร', text)

    temp = re.sub('[0-9]', '', text)
    text = ' '.join(temp.split())

    temp = re.sub('[!@#$%^&*,?+()-/\.:;"…><_]', '', text)
    text = ' '.join(temp.split())

    # Emoji Remover
    temp = demoji.replace(text,'')
    text = ' '.join(temp.split())

    # English Remover
    temp = re.sub('[A-Za-z]', '', text)
    text = ' '.join(temp.split()) 

    rmv = text

    ## Word Split
    newmm = tokenizer_newmm.word_tokenize(rmv)

    ## Stop Word Remover
    newmm = [j for j in newmm if j not in new_stopwords]

    ## Thai Letter Only
    newmm = re.sub('[^ก-๙0-9 ]', '', ' '.join(newmm))

    return newmm

def classifing(text):
    tfidf_vec_all = TfidfVectorizer(analyzer=lambda x:x.split(' '))

    X_all = tfidf_vec_all.fit_transform([text])

    display_tfidf_all = pd.DataFrame(X_all.toarray(), columns = tfidf_vec_all.get_feature_names_out())

    table_all = pd.read_csv('./ml/keywords.csv', sep=';')

    param = pd.DataFrame(columns=list(table_all['Word']))

    
    df = pd.concat([param, display_tfidf_all], join='outer')

    df1 = df[list(table_all['Word'])]
    df1 = df1.fillna(0)

    loaded_model = pickle.load(open('.\ml\clf_xgb_model.sav', 'rb'))
    result = loaded_model.predict(df1)[0]
    prob = loaded_model.predict_proba(df1)[0][result] * 100
    return result, prob

def main(text):
    classlist = ['MyMo', 'บัตรและสลากออมสิน', 'สินเชื่อ', 'อื่น ๆ', 'เงินฝาก']
    text_token = tokenizing(text)
    text_class, prob = classifing(text_token)
    data = [text, int(text_class+1), classlist[text_class], prob]
    return data


if __name__ == '__main__':
    text = sys.argv[1]
    main(text)