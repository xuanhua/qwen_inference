from typing import (
  Sequence
)
from collections import (
  defaultdict
)
from sklearn.feature_extraction.text import (
  TfidfVectorizer
)
from sklearn.cluster import AgglomerativeClustering
import json



def cut_comments_to_sub_comments_by_separator(comment_list: Sequence[str],
                              separators: Sequence[str]):
  """
  Split comments into pieces by separators.
  """
  piece_list = []
  for comment in comment_list:
    sub_cmt_list = [cmt for cmt in comment.split(separators)]
    piece_list += sub_cmt_list
  return piece_list

def drop_stopwords(comment_list: Sequence[str]):
  """
  Drop stopwords from comments.
  """
  pass

def cluster_comments(comment_list: Sequence[str],
                     n_clusters=20):
  """
  Perform clustering of single turn utterances based on their content.
  """
  # Create a TF-IDF matrix of the utterances
  vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(1,3), # Use unigrams, bigrams and trigrams
  )
  X = vectorizer.fit_transform(comment_list).todense()

  # Perform clustering using cosine similarity as metric
  clusterer = AgglomerativeClustering(n_clusters=n_clusters)
  clusterer.fit(X)

  return clusterer.labels_, vectorizer.get_feature_names()


import pandas as pd
def load_columns_from_csv(csv_path:str, column_names:Sequence[str]):
  """
  """
  df = pd.read_csv(csv_path,
                   usecols=column_names,
                   header="infer")
  return df[column_names].values.tolist()

import argparse
def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv-path', type=str, required=True)
  parser.add_argument('--column-names', nargs='+', type=str, required=True, default=None)
  parser.add_argument('--num-clusters', type=int, required=False, default=20)
  args = parser.parse_args()
  return args

if __name__ == '__main__':

  args = get_args()
  comments = load_columns_from_csv(args.csv_path, args.column_names)
  comments = [cmt[0] for cmt in comments if isinstance(cmt, list) and len(cmt) > 0] # Flatten the list of lists

  labels, _ = cluster_comments(comments,
                               n_clusters=args.num_clusters)

  print("Total number of clusters: ", len(set(labels)))
  label2cmts  = defaultdict(list)
  assert len(comments) == len(labels), "Utterances and labels should have the same length"
  for cmt, label in zip(comments, labels):
    label2cmts[int(label)].append(cmt)
  print(json.dumps(label2cmts, indent=2, ensure_ascii=False))