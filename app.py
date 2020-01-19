from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import re
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
import string
import unidecode
import json
from bs4 import BeautifulSoup
import requests

import spacy

import torch
import numpy as np
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=3)
device = torch.device("cpu")
model.load_state_dict(torch.load("./roberta.pt", map_location=device))
model.to(device)
model.eval()
tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

LANGUAGE = 'english'
SENTENCES_COUNT = 15
clause = re.compile("(will)|(agree)|(must)|(responsib)|(waive)|(lawsuit)|(modify)|(intellec)", re.IGNORECASE)
host_reg = re.compile('(?:http.*://)?(?P<host>[^:/ ]+).?(?P<port>[0-9]*).*')
terms_page = re.compile(r"(terms *(((and|&)? *conditions)|((of)? ?(service|use))))", re.IGNORECASE)
HTML_OPEN = "<div id='mainPopup'>"
YOU_AGREE_HEADER = "<div id='youAgree' class='header'>What You Agree</div>"
THEY_AGREE_HEADER = "<div id='theyAgree' class='header'>What They Agree</div>"
OTHER_HEADER = "<div id='other' class='header'>Other Clauses</div>"

nlp = spacy.load("en_core_web_sm")

VERBS = ["responsible", "collect", "protect", "control", "sell", "access", "share", "disclosure", "collection", "notify", "consent", "inform", "withdraw", "provide", "remain", "agree", "object", "obtain"]
KEYWORDS = ["third- party", "third party", "face recognition", "personal data", "personal privacy", "contractual obligations"]

def prepare_for_regex(input_str, delimiter="."):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', input_str)
    pure_sentences = cleantext.strip().split(delimiter)
    if pure_sentences[-1] == "":
        pure_sentences.pop()

    # Remove leading/trailing whitespace and end with period.
    for i, x in enumerate(pure_sentences):
        pure_sentences[i] = x.strip() + "."

    clean_sentences = []
    for x in pure_sentences:
        clean_sentences.append(' '.join(x.lower().translate(str.maketrans({key: None for key in string.punctuation})).split()))
    return clean_sentences, pure_sentences

app = Flask(__name__)
CORS(app)

@app.route('/', methods=["GET","POST"])
def main():
    if request.method == "POST":
        if 'link' in request.json:
            link = request.json['link']
            r = requests.get(link)
            html_soup = BeautifulSoup(r.text, "html.parser")
            text = html_soup.text
        elif 'text' in request.json:
            text = request.json['text']
    else:
        if 'text' in request.args:
            text = request.args['text']
        elif 'link' in request.args:
            link = request.args['link']
            r = requests.get(link)
            html_soup = BeautifulSoup(r.text, "html.parser")
            text = html_soup.text
        else:
            return "Please provide text"
    return process_data(text)

def process_data(text):
    text_data = unidecode.unidecode(text)
    clean_list, pure_list = prepare_for_regex(text_data)

    data_to_summarize = []
    for clean, pure in zip(clean_list, pure_list):
        if re.findall(clause, clean):
            data_to_summarize.append(pure)
    text_data = " ".join(data_to_summarize)
    parser = PlaintextParser(text_data, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = TextRankSummarizer(stemmer)

    summary = summarizer(parser.document, SENTENCES_COUNT)
    sentences = []
    for sentence in summary:
        skip = False
        for punct in ["[","]","{","}","=","+","_","|","<",">","^"]:
            if punct in str(sentence):
                skip = True
        if skip:
            continue
        if str(sentence)[-1] == "." and len(str(sentence))<500:
            try:
                int(str(sentence)[0])
                sentence = str(sentence)[1:].strip()     
            except:
                sentence = str(sentence).strip()
            sent = nlp(sentence)
            sentence = ""
            entities = []
            for token in sent:
                if sentence and token.text.strip() not in string.punctuation:
                    sentence += " "
                if token.text.lower() in VERBS:
                    sentence += '''<mark class="entity" style="background: #ffffb3; padding: 0.2em 0.2em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">{}</mark>'''.format(token.text)
                    entities.append(token.text)
                else:
                    sentence += token.text
            sentence = sentence[:-1] + ". "
            for ent in sent.ents:
                if ent.text == "IP":
                    continue
                if ent.text not in entities and ent.label_ == "ORG" or ent.text.lower()=="stripe":
                    sentence = sentence.replace(ent.text, '''<mark class="entity" style="background: #ccffcc; padding: 0.2em 0.2em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">{}</mark> '''.format(ent.text))
                    entities.append(ent.text)
            for term in KEYWORDS:
                if term in entities:
                    continue
                case_insensitive = re.compile(re.escape(term), re.IGNORECASE)
                sentence = case_insensitive.sub('''<mark class="entity" style="background: #b3d9ff; padding: 0.2em 0.2em;; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">{}</mark>'''.format(term), sentence)
            sentences.append({
                "text": sentence,
                "rating": get_sentiment(sentence)
            })
    readability_score, complexity_score = get_readability(text_data)
    return jsonify({
        "summary_sentences":sentences,
        "readability_score": readability_score,
        "complexity_score": complexity_score
    })
    
def get_sentiment(sentence):
    tokens_a = tokenizer.tokenize(sentence)
    if len(tokens_a)>100:
        tokens_a = tokens_a[:100]
    one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])
    inp = torch.tensor(one_token).long().reshape(1,-1)
    pred = model(inp.to(device), attention_mask=(np.logical_not(inp==0)).to(device), labels=None)
    scores = F.softmax(pred[0], dim=1)[0].cpu().detach().numpy()
    sentiment = scores[2] + 0.5*scores[1]
    if sentiment < 0.4:
        return -1
    elif sentiment > 0.7:
        return 1
    return 0

def get_readability(text_data):
    headers = {
        'X-RapidAPI-Host': 'ipeirotis-readability-metrics.p.rapidapi.com',
        'X-RapidAPI-Key': 'f9ubYNvChSmsh7ziUqOiramFPcBnp1UxuM2jsnCqrPp4oiO7Dw'
    }
    r = requests.post("https://ipeirotis-readability-metrics.p.rapidapi.com/getReadabilityMetrics", headers=headers, data={"text":text_data})
    response = r.json()
    try:
        return 100-response['FLESCH_KINCAID'], response['COMPLEXWORDS']/response['WORDS']
    except:
        return 0, 0

@app.route('/text')
def text_box():
    return render_template('index.html')

@app.route('/submit_data')
def submit_data():
    text = request.args.get('value')
    return process_data(text)

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=80)