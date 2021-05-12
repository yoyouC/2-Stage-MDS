from tools.data_loader import load_dataset, save_to_json
import csv, math, json

def read_tsv(tsv_path, columns, is_int):
    ret = []

    with open(tsv_path) as tsv:
        tsv = csv.reader(tsv, delimiter="\t")
        for line in tsv:
            item = {}
            for i, column in enumerate(columns):
                if column == 'date':
                    item[column] = line[i].replace('T00:00:00.0000000', '')
                    continue
                if not is_int[i]:
                    item[column] = line[i]
                else:
                    item[column] = int(line[i])
            ret.append(item)
    
    return ret

def add_to_dataset(dataset_path, extra_info):
    dataset = load_dataset(dataset_path)
    for item in dataset:
        mid = int(item['mid'])
        info = binary_search(mid, extra_info)
        if info == None:
            continue
        for (key, value) in info.items():
            item[key] = value
    return dataset

def add_to_dataset_reformat(dataset_path, extra_info):
    dataset = load_dataset(dataset_path)
    queries = dataset['queries']
    references = dataset['references']

    for item in queries:
        if item['mid'] == '':
            references.remove(item)
        mid = int(item['mid'])
        info = binary_search(mid, extra_info)
        if info == None:
            continue
        for (key, value) in info.items():
            if key == 'mid':
                continue
            item[key] = value

    for item in references:
        if item['mid'] == '':
            references.remove(item)
            continue
        mid = int(item['mid'])
        info = binary_search(mid, extra_info)
        if info == None:
            continue
        for (key, value) in info.items():
            if key == 'mid':
                continue
            item[key] = value

    return dataset

def add_rec_to_dataset_reformat(dataset_path, extra_info):
    dataset = load_dataset(dataset_path)
    references = dataset['references']

    for item in references:
        mid = int(item['mid'])
        rec_paper_id = []
        info = binary_search_list(mid, extra_info)

        if info == None:
            continue
        
        for dic in info:
            rec_paper_id.append(dic['rec_mid'])
        
        item['rec_mid'] = rec_paper_id

    return dataset

def binary_search(paper_id, l):
    lower = 0
    higher = len(l) - 1

    while higher >= lower:
        mid = math.floor((higher + lower) / 2)
        if l[mid]['mid'] > paper_id:
            higher = mid - 1
        elif l[mid]['mid'] < paper_id:
            lower = mid + 1
        else:
            return l[mid]
    
    return None

def binary_search_list(paper_id, l):
    lower = 0
    higher = len(l) - 1
    found = None
    ret = []

    while higher >= lower:
        mid = math.floor((higher + lower) / 2)
        if l[mid]['mid'] > paper_id:
            higher = mid - 1
        elif l[mid]['mid'] < paper_id:
            lower = mid + 1
        else:
            found = mid
            break

    if found == None:
        return None

    while l[found]['mid'] == paper_id:
        found -= 1

    found += 1

    while l[found]['mid'] == paper_id:
        ret.append(l[found])
        found += 1
    
    return ret

# out = read_tsv('out.tsv', 
#         ['mid', 'rank', 'year', 'date', 'paper_title', 'publisher', 'reference_count', 'citation_count', 'estimated_citation'],
#         [True, True, True, False, False, False, True, True, True])

# dataset = add_to_dataset_reformat('dataset/all_reformat', out)
# save_to_json(dataset, 'dataset/all_reformat_extra')
# dataset = load_dataset('dataset/all_reformat_extra')
# print(dataset['queries'][0])
# print(dataset['references'][0])

# paper_recommandation = read_tsv('dataset/RecommendPaper.tsv', ['mid', 'rec_mid', 'score'], [True, True, False])
# dataset = add_rec_to_dataset_reformat('dataset/all_reformat_extra', paper_recommandation)
# save_to_json(dataset, 'dataset/all_reformat_extra_plus')
dataset = load_dataset('dataset/all_reformat_extra_plus')
# print(dataset['queries'][0])
print(json.loads(str(dataset['references'][0]['rec_mid'])))
