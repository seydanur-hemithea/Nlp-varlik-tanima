#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

# Sadece gerekli olan Türkçe ve NER modelini indiriyoruz (RAM tasarrufu)
python -c "import stanza; stanza.download('tr', processors='tokenize,ner')"
