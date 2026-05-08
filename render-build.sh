#!/usr/bin/env bash
set -o errexit

# Bağımlılıkları kur
pip install -r requirements.txt

# Türkçe modelini güvenli yoldan indir
python -m spacy download tr_core_news_lg
