import gzip
import json


def reformat_multi_xscience(src_filepath, tgt_filepath):
    with gzip.GzipFile(filename=src_filepath, mode='r') as reader:
        json_bytes = reader.read() 
    
    json_str = json_bytes.decode('utf-8')            
    src_dataset = json.loads(json_str)

    q_set = {}
    queries = []
    references = [] 

    for src_data in src_dataset:
        references_mid = []

        for _, (_, reference) in enumerate(src_data['ref_abstract'].items()):
            if not reference['mid'] == '':
                references_mid.append(reference['mid'])
            if not reference in references:
                references.append(reference)
        
        if src_data['mid'] not in q_set:
            q_set[src_data['mid']] = len(queries)
            queries.append({'aid': src_data['aid'], 
            'mid': src_data['mid'], 
            'abstract': src_data['abstract'],
            'related_work': [src_data['related_work']],
            'references_mid': references_mid})
        else:
            for ref_mid in references_mid:
                if ref_mid not in queries[q_set[src_data['mid']]]['references_mid']:
                    queries[q_set[src_data['mid']]]['references_mid'].append(ref_mid)
            queries[q_set[src_data['mid']]]['related_work'].append(src_data['related_work'])

    tgt_dataset = {'queries': queries, 'references': references}

    json_str = json.dumps(tgt_dataset) + "\n"
    json_bytes = json_str.encode('utf-8')

    with gzip.GzipFile(tgt_filepath, 'w') as fout:
        fout.write(json_bytes)    


def reformat_for_eval(src_filepath, tgt_filepath):
    with gzip.GzipFile(filename=src_filepath, mode='r') as reader:
        json_bytes = reader.read() 
    
    json_str = json_bytes.decode('utf-8')            
    src_dataset = json.loads(json_str)

    papers = []
    paper_index = {}

    for src_data in src_dataset:
        if src_data['mid'] not in paper_index:
            paper_index[src_data['mid']] = len(papers)
            papers.append({'aid': src_data['aid'], 
            'mid': src_data['mid'], 
            'abstract': src_data['abstract'],
            'related_work': src_data['related_work'],
            'ref_abstract': [src_data['ref_abstract']]})
        else:
            papers[paper_index[src_data['mid']]]['ref_abstract'].append(src_data['ref_abstract'])
            papers[paper_index[src_data['mid']]]['related_work'] += ' ' + src_data['related_work']

    tgt_dataset = papers

    json_str = json.dumps(tgt_dataset) + "\n"
    json_bytes = json_str.encode('utf-8')

    with gzip.GzipFile(tgt_filepath, 'w') as fout:
        fout.write(json_bytes) 

def reformat_for_cluster(src_filepath, tgt_filepath):
    with gzip.GzipFile(filename=src_filepath, mode='r') as reader:
        json_bytes = reader.read() 
    
    json_str = json_bytes.decode('utf-8')            
    src_dataset = json.loads(json_str)
    
    q_set = {}
    queries = []

    for src_data in src_dataset:
        ref_abstracts = []

        for _, (_, reference) in enumerate(src_data['ref_abstract'].items()):
            if not reference['abstract'] == '':
                ref_abstracts.append(reference['abstract'])
        
        if src_data['mid'] not in q_set:
            q_set[src_data['mid']] = len(queries)
            queries.append({ 
            'mid': src_data['mid'], 
            'abstract': src_data['abstract'],
            'ref_abstract': [ref_abstracts]})
        else:
            queries[q_set[src_data['mid']]]['ref_abstract'].append(ref_abstracts)

    tgt_dataset = queries

    json_str = json.dumps(tgt_dataset) + "\n"
    json_bytes = json_str.encode('utf-8')

    with gzip.GzipFile(tgt_filepath, 'w') as fout:
        fout.write(json_bytes)    

src_dir = "./dataset/Multi-XScience-cluster/"
tgt_dir = "./dataset/Multi-XScience-cluster/"
files = ['train.json.gz']

for file in files:
    src_filepath = src_dir + file
    tgt_filepath = tgt_dir + file
    reformat_for_cluster(src_filepath, tgt_filepath)
