import gzip
import json
import csv

def load_dataset(filepath):
    with gzip.GzipFile(filename=filepath, mode='r') as reader:
        json_bytes = reader.read() 

    json_str = json_bytes.decode('utf-8')            
    dataset = json.loads(json_str)

    return dataset

def get_mid_from_dataset(filepath, mids):

    dataset = load_dataset(filepath)
    for reference in dataset['refernces']:
        if not reference['mid'] in mids:
            mids.append(reference['mid'])
    for query in dataset['queries']:
        if not query['mid'] in mids:
            mids.append(query['mid'])

def read_tsv(filepath):
    file = open(filepath)
    read_tsv = csv.reader(file, delimiter="\t")
    result = []
    for ele in read_tsv:
        result.append(ele)
    return result

if __name__ == "__main__":
    # mids = []
    # get_mid_from_dataset("./dataset/Multi-XScience-reformat/test.json.gz", mids)
    # get_mid_from_dataset("./dataset/Multi-XScience-reformat/val.json.gz", mids)
    # get_mid_from_dataset("./dataset/Multi-XScience-reformat/train.json.gz", mids)
    # print(len(mids))
    # mids.sort(key=lambda ele: int(ele))
    # with open("MultiXSciencePaperId.txt", "w", encoding='utf-8') as record_file:
    #     for mid in mids:
    #         if mid.strip() != '':
                # record_file.write(" +" + mid.strip() + " " + "\n")

    # dataset = load_dataset('./dataset/PaperReferences.json.gz')
    # print(len(dataset))
    
    # paperIds = read_tsv('./dataset/MultiXSciencePaperId.txt')
    # paperIds_with_ref = read_tsv('./dataset/MultiXSciencePaperReferences.tsv')
    
    # print(paperIds_with_ref[0][0])
    # print(paperIds[0][0])
    # index = 0
    # missing = []
    # current_id = None
    # for ele in paperIds_with_ref:
    #     id = ele[0]
    #     if id == current_id:
    #         continue

    #     current_id = id
    #     if current_id != paperIds[index][0]:
    #         while current_id != paperIds[index][0]:
    #             missing.append(paperIds[index][0])
    #             index += 1
    #         index += 1
    #     else:
    #         index += 1
    
    # print(len(missing))

    data = load_dataset("./dataset/Multi-XScience/test.json.gz")
    print(data[1])