#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

from docembedder.simspec import SimulationSpecification
from docembedder.models import TfidfEmbedder
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.models.doc2vec import D2VEmbedder
from docembedder.models import CountVecEmbedder
from docembedder.models import BERTEmbedder

from docembedder.utils import run_models
from docembedder.pretrained_run import pretrained_run_models
import datetime





def parse_arguments():
    parser = ArgumentParser(
        prog="create_embeddings.py",
        description="Create embeddings")
    parser.add_argument("--patent_dir", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--cpc_fp", required=True)
    # parser.add_argument("--year_start", required=True, type=int)
    # parser.add_argument("--year_end", required=True, type=int)
    return parser.parse_args()


def compute_embeddings(patent_dir, output_fp, cpc_fp):
    year_start = 1838
    year_end = 1951
    # set simulation specification
    sim_spec = SimulationSpecification(
        year_start=year_start,
        year_end=year_end,
        window_size=21,
        window_shift=1,
    )
    
    model_cv = {
    "countvec": CountVecEmbedder(method='sigmoid')
    }
    prep_cv = {
    "prep-countvec": Preprocessor(keep_caps=False, keep_start_section=False, remove_non_alpha=True)
    }

    model_tfidf = {
        "tfidf": TfidfEmbedder(ngram_max=1,stop_words='english',stem=True, norm='l1', sublinear_tf=True, min_df=6, max_df=0.665461)
    }
    prep_tfidf = {
        "prep-tfidf": Preprocessor(keep_caps=True, keep_start_section=True, remove_non_alpha=True),
    }
    
    model_doc2vec = {
        "doc2vec": D2VEmbedder(epoch=9, min_count=7, vector_size=101)
    }
    prep_doc2vec = {
        "prep-doc2vec": Preprocessor(keep_caps=False, keep_start_section=True, remove_non_alpha=False)
    }
    
    model_bert = {
    "bert":BERTEmbedder(pretrained_model='AI-Growth-Lab/PatentSBERTa')
    }
    prep_bert = {
     "prep-bert": Preprocessor(keep_caps=True, keep_start_section=True, remove_non_alpha=True)
    }
    
    


    for year in range(year_start, year_end):
        if not (patent_dir / f"{year}.xz").is_file():
            raise ValueError(f"Please download patent file {year}.xz and put it in"
                             f"the right directory ({patent_dir})")

    # run_models(prep_cv, model_cv, sim_spec, patent_dir, output_fp, cpc_fp, n_jobs=4)
    # print('cv emdeddings are calculated...')
    # run_models(prep_tfidf, model_tfidf, sim_spec, patent_dir, output_fp, cpc_fp, n_jobs=4)
    # print('tfidf emdeddings are calculated...')
    run_models(prep_doc2vec, model_doc2vec, sim_spec, patent_dir, output_fp, cpc_fp, n_jobs=2)
    print('doc2vec emdeddings are calculated...')
    # pretrained_run_models(prep_bert, model_bert, sim_spec, patent_dir, output_fp, cpc_fp)
    # print('bert emdeddings are calculated...')



if __name__ == "__main__":
    start = datetime.datetime.now()
    print(start)
    args = parse_arguments()
    compute_embeddings(Path(args.patent_dir), Path(args.embedding),
                       Path(args.cpc_fp))
    end = datetime.datetime.now()
    print(end)

    duration = end - start
    print(duration)
