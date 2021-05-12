from retriever.retriever import LuceneRetriever
from retriever.retriever import EntityEncodingRetriever
from tools.data_loader import load_dataset, save_to_json
from sklearn import metrics
import clustering.tfidf
import clustering.roberta

def eval_TF_IDF_retriever():
    retriever = LuceneRetriever()
    dataset = load_dataset('dataset/all_reformat_extra')
    queries = dataset['queries']
    ground_truth_mid = [query['references_mid'] for query in dataset['queries']]

    print(retriever.evaluate_performance(queries[:100], ground_truth_mid[:100], k=100, with_time=True, use_title=True))

def eval_Entity_Encodig_retriever():
    retriever = EntityEncodingRetriever('encodings/hidden_queries', 'encodings/hidden_references')
    dataset = load_dataset('dataset/all_reformat_extra')
    queries = dataset['queries']
    ground_truth_mid = [query['references_mid'] for query in dataset['queries']]

    print(retriever.evaluate_performance([q['mid'] for q in queries][:20], ground_truth_mid[:20], k=100))

def eval_Roberta_Clustering(encoding='tf-idf', method='k-means'):
    dataset = load_dataset('dataset/Multi-XScience-cluster/train.json.gz')
    dataset = [item['ref_abstract'] for item in dataset]
    document_clusters = []
    count = 0

    for c in dataset:
        if len(c) > 1:
            document_clusters.append(c)
            count += 1
        if count == 60:
            break
    
    avg_rand_index = 0
    avg_adjusted_rand_index = 0
    avg_AMI = 0
    avg_v_measure = 0
    counter = 0
    for clusters in document_clusters:
        count = 0
        documents = []
        ture_label = []
        paragraph_num = len(clusters)

        for c in clusters:
            for doc in c:
                documents.append(doc)
                ture_label.append(count)
            count += 1
        
        if encoding == 'tf-idf':
            _, pred_label = clustering.tfidf.cluster_doc(documents, n=paragraph_num, method=method)
        else:
            _, pred_label = clustering.roberta.cluster_doc(documents, n=paragraph_num, method=method)

        avg_rand_index += metrics.rand_score(ture_label, pred_label)
        avg_adjusted_rand_index += metrics.adjusted_rand_score(ture_label, pred_label)
        avg_AMI += metrics.adjusted_mutual_info_score(ture_label, pred_label)
        avg_v_measure += metrics.v_measure_score(ture_label, pred_label)

        counter += 1
        print(counter)
        # print(pred_label)
        # print(ture_label)

    print("RI:", avg_rand_index/60)
    print("ARI:", avg_adjusted_rand_index/60)
    print("AMI:", avg_AMI/60)
    print("VM:", avg_v_measure/60)
    

eval_Roberta_Clustering(encoding='roberta', method='AC')
# eval_TF_IDF_retriever()
# eval_Entity_Encodig_retriever()y