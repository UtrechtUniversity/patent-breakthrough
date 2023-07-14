#!/bin/bash

./init_windows.py --settings test-settings.json
./create_embeddings.py --patent_dir ../data/raw/unprocessed/ --window_fp ../data/output/windows/1838-1842.npy --output_fp ../data/output/modelres/tfidf/1838-1842/embedding.npy --model_fp tfidf.json
./create_novelty_impact.py --embedding_fp ../data/output/modelres/tfidf/1838-1842/embedding.npy --output_csv ../data/output/modelres/tfidf/1838-1842/impact_novelty.csv --window_fp ../data/output/windows/1838-1842.npy
