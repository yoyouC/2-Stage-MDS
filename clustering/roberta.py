import torch
from sklearn.cluster import KMeans
from transformers import RobertaTokenizer, RobertaModel
from sklearn.cluster import AgglomerativeClustering

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

def encode_doc(doc_str):
    tokens = tokenizer(doc_str, return_tensors="pt")
    outputs = model(**tokens)
    last_hidden_states = torch.squeeze(outputs.last_hidden_state)
    doc_vec = torch.mean(last_hidden_states, dim=0)
    return doc_vec.detach().numpy()

def cluster_doc(doc_strs, n, method='k-means'):
    result = []
    doc_vecs = [encode_doc(doc_str) for doc_str in doc_strs]

    if method == 'k-means':
        labels = KMeans(n_clusters=n, random_state=0).fit(doc_vecs).labels_
    else:
        labels = AgglomerativeClustering(n_clusters=n).fit(doc_vecs).labels_

    for i in range(n):
        result.append([tup[0] for tup in zip(doc_strs, labels) if tup[1] == i])
    return result, labels