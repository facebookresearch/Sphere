# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import torch
import zlib

from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder

from distributed_faiss.client import IndexClient


class RetrievalClient:
    def __init__(self, tokenizer, encoder, discovery_config, index_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(tokenizer)
        self.encoder = DPRQuestionEncoder.from_pretrained(encoder).to(self.device)

        # connectind to the distributed-faiss server
        self.index_client = IndexClient(discovery_config)
        self.index_id = index_id
        print("Loading remote index")
        self.index_client.load_index(args.index_id, force_reload=False)
        print("Done loading index")

    def encode_query(self, questions):
        inputs = self.tokenizer.batch_encode_plus(
            questions,
            return_tensors="pt",
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
        )["input_ids"].to(self.device)
        # See https://github.com/facebookresearch/DPR/blob/multi_task_training/dpr/models/hf_models.py#L307
        # to justify the line below
        inputs[:, -1] = self.tokenizer.sep_token_id
        with torch.no_grad():
            question_tensors = self.encoder(inputs)[0]
            return question_tensors

    def get_top_docs(
        self,
        query_vectors: np.array,
        top_docs: int = 100,
        use_l2_conversion: bool = True,
    ):
        results = []
        # Sphere index is build for l2 distance - extra dim required for compatilibilty with dot product, more details
        # in https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-do-max-inner-product-search-on-indexes-that-support-only-l2
        if use_l2_conversion:
            aux_dim = np.zeros(len(query_vectors), dtype="float32")
            query_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
        scores, metas = self.index_client.search_with_filter(query_vectors, top_docs, self.index_id, 3, True)
        results.extend([(metas[q], scores[q]) for q in range(len(scores))])
        return results


def main(args):
    retrieval_client = RetrievalClient(args.tokenizer, args.encoder, args.discovery_config, args.index_id)

    while True:
        print("Type the query...")
        query = input()
        questions_tensor = retrieval_client.encode_query([query])
        docs_and_scores = retrieval_client.get_top_docs(questions_tensor.cpu().numpy(), top_docs=args.k)
        for doc_ids, doc_scores in docs_and_scores:
            for i, t in enumerate(doc_ids):
                print("Doc id: ", t[0])
                print("Title:", zlib.decompress(t[2]).decode())
                print("Text:", zlib.decompress(t[1]).decode())
                print("Retrieval score:", float(doc_scores[i]))
                print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--discovery-config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="facebook/dpr-question_encoder-single-nq-base")
    parser.add_argument("--index-id", type=str, default="dense")
    parser.add_argument("-k", type=int, default=5, help="number of docs to retrieve")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
