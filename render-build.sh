#!/usr/bin/env bash
set -o errexit

# 1. Temel paketleri kur
pip install -r requirements.txt

# 2. Medium modelini güvenli kaynaktan indir
python -m spacy download tr_core_news_md
