#!/usr/bin/env bash
set -o errexit

# 1. Bağımlılıkları kur
pip install -r requirements.txt

# 2. Türkçe modelini doğrudan pip üzerinden GitHub reposundan çek (Linkten daha güvenli)
pip install https://github.com/explosion/spacy-models/releases/download/tr_core_news_lg-3.7.0/tr_core_news_lg-3.7.0-py3-none-any.whl
