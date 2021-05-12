import lucene
import numpy as np
import json, networkx
import torch
import torch.nn as nn
 
from java.io import File
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, TextField, Field, StringField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory, RAMDirectory
from org.apache.lucene.util import Version

from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity, ClassicSimilarity
from org.apache.lucene.index import IndexReader
from org.apache.lucene.queryparser.classic import QueryParser

from data_loader import load_dataset

lucene.initVM()

analyzer = StandardAnalyzer()
path = Paths.get("retriever/runtime/index/")
directory = SimpleFSDirectory.open(path)
reader = DirectoryReader.open(directory)
searcher = IndexSearcher(reader)
searcher.setSimilarity(BM25Similarity())

dataset = load_dataset("dataset/Multi-XScience-reformat/train.json.gz")
queries = dataset['queries']

def retrieve_top_k(query, k=10):
    result = []

    q = QueryParser("abstract", analyzer).parse(QueryParser.escape(query))
    docs = searcher.search(q, k)
    hits = docs.scoreDocs

    for hit in hits:
        docId = hit.doc
        doc = searcher.doc(docId)
        result.append(doc['abstract'])
    
    return result

def cal_recall_precision(predict, ground_truth):
    hits = len([p for p in predict[:len(ground_truth)] if p in ground_truth])
    recall = hits/len(ground_truth)
    precision = hits/len(predict)
    return recall, precision

def cal_hit_at_k_recall(predict, ground_truth):
    ground_truth_len = len(ground_truth)
    hits = 0
    hit_at_k = []
    for p in predict:
        if p in ground_truth:
            hits += 1
        hit_at_k.append(hits/ground_truth_len)

    return hit_at_k

def compare_date(a, b):
    a = a.split('-')
    b = b.split('-')

    if int(a[0]) >= int(b[0]):
        if int(a[0]) > int(b[0]):
            return True
        if int(a[1]) > int(b[1]):
            return True
    
    return False
    

class LuceneRetriever():
    def __init__(self, index_dir="retriever/runtime/index_all_extra_plus/"):
        lucene.initVM()

        path = Paths.get(index_dir)
        self.analyzer = StandardAnalyzer()
        self.directory = SimpleFSDirectory.open(path)
        self.reader = DirectoryReader.open(self.directory)
        self.searcher = IndexSearcher(self.reader)
        self.searcher.setSimilarity(ClassicSimilarity())
    
    def gen_query(self, query_str):
        query = QueryParser.escape(query_str.lower())
        query = QueryParser("abstract", self.analyzer).parse(query)
        return query

    def retrieve_top_k_abstracts(self, query, k=10):
        abstracts = []

        query = self.gen_query(query)
        docs = self.searcher.search(query, 10)
        hits = docs.scoreDocs

        for hit in hits:
            docId = hit.doc
            doc = searcher.doc(docId)
            abstracts.append(doc['abstract'])
        
        return abstracts
    
    def retrieve_top_k_mid(self, query, k=10, use_title=False):
        if use_title and 'paper_title' in query:
            query = self.gen_query(query['paper_title'] + ' ' + query['paper_title'] + ' ' + query['abstract'])
        else:
            query = self.gen_query(query['abstract'])
        docs = self.searcher.search(query, k)
        hits = docs.scoreDocs

        predict = []
        for hit in hits:
            docId = hit.doc
            doc = self.searcher.doc(docId)
            predict.append(doc['mid'])

        return predict

    def retrieve_top_k_with_time(self, query, k=10, use_title=False, mid=True):
        if use_title and 'paper_title' in query:
            q = self.gen_query(query['paper_title'] + ' ' + query['paper_title'] + ' ' + query['abstract'])
        else:
            q = self.gen_query(query['abstract'])

        docs = self.searcher.search(q, 1000)
        hits = docs.scoreDocs

        predict = []
        for hit in hits:
            docId = hit.doc
            doc = self.searcher.doc(docId)
            predict.append(doc)
        
        for p in predict:
            if p['date'] != 'unknown':
                if not compare_date(query['date'], p['date']):
                    predict.remove(p)
        
        if mid:
            return [p['mid'] for p in predict][:k]
        else:
            return [p for p in predict][:k]
    
    def retrieve_top_k_with_rec(self, query, k):
        predict = self.retrieve_top_k_with_time(query, k, True, False)
        predict_mids = [p['mid'] for p in predict]
        G = networkx.Graph()
        print(predict_mids)
        for p in predict:
            if p['rec_mid'] != 'unknown':
                rec_mids = json.loads(p['rec_mid'])
                for mid in rec_mids:
                    if str(mid) in predict_mids:
                        G.add_edge(p['mid'], str(mid))

        cliques = list(networkx.find_cliques(G))
        print(cliques)
        selected = []

        # for c in cliques:
        #     for node in c:
        #         if node not in selected:
        #             selected.append(node)
        
        exit()
        
        # score = networkx.pagerank(G, alpha=0.8)
        # selected = []
        # for (key, item) in score.items():
        #     selected.append((key, item))
        # selected.sort(key=lambda x:x[1])
        # selected = [s[0] for s in selected][:5]
        predict_mids = [p for p in predict_mids if p not in selected]
        selected.extend(predict_mids)

        return selected


    def evaluate_performance(self, queries, ground_truths, k=10, with_time=False, use_title=False):
        sum_recall = 0
        sum_precision = 0
        hit_at_ks = []

        for (query, ground_truth) in zip(queries,ground_truths):
            predict = None

            if with_time and 'date' in query:
                predict = self.retrieve_top_k_with_time(query, k, use_title)
            else:
                predict = self.retrieve_top_k_mid(query, k, use_title)

            recall, precision = cal_recall_precision(predict, ground_truth)
            sum_recall += recall
            sum_precision += precision

            hit_at_ks.append(cal_hit_at_k_recall(predict, ground_truth))

            # print(sum_recall, sum_precision)

        return np.mean(hit_at_ks, axis=0), sum_recall/len(queries)

class EntityEncodingRetriever():

    def __init__(self, encoded_qry_path, encoded_ref_path):
        self.query_vec = torch.load(encoded_qry_path)
        self.refer_vec = torch.load(encoded_ref_path)
    
    def evaluate_performance(self, query_mids, ground_truths, k=10):
        sum_recall = 0
        sum_precision = 0
        hit_at_ks = []

        for (mid, ground_truth) in zip(query_mids, ground_truths):

            predict = self.retrieve_top_k(mid, k)
            print(predict)
            print(ground_truth)
            recall, precision = cal_recall_precision(predict, ground_truth)
            sum_recall += recall
            sum_precision += precision

            hit_at_ks.append(cal_hit_at_k_recall(predict, ground_truth))

            # print(sum_recall, sum_precision)

        return np.mean(hit_at_ks, axis=0)
    
    def retrieve_top_k(self, mid, k):
        sims = []
        q_v = self.query_vec[mid].double() - 128
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        for (mid, r_v) in self.refer_vec.items():
            r_v = r_v.double() - 128
            sim = cos(q_v, r_v)
            sims.append((mid, sim))
        return self.find_top_k(sims, k)
        

        
    def find_top_k(self, sim_list, k):
        sim_list.sort(key=lambda x : x[1])
        return [l[0] for l in sim_list[:k]]