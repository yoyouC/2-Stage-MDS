from tools.data_loader import load_dataset
from nltk import word_tokenize
from re import sub

def vocab_from_dataset(src_filepath, des_filepath):
    vocab_set = {}
    dataset = load_dataset(src_filepath)

    for item in dataset:
        abstract = item['abstract'].lower()
        related_work = sub("@cite_\S*", "@cite", item['related_work'].lower())
        ref_abstracts = item['ref_abstract']

        add_vocabs_to_set(word_tokenize(abstract), vocab_set)
        add_vocabs_to_set(word_tokenize(related_work), vocab_set)
        for ref_abstract in ref_abstracts:
            add_vocabs_to_set(word_tokenize(ref_abstract.lower()), vocab_set)

    vocab_list = [(k, v) for k, v in vocab_set.items()] 
    vocab_list.sort(key= lambda tuple: tuple[1])
    
    with open(des_filepath, "w") as des:
        for (vocab, freq) in vocab_list[::-1]:
            des.write(vocab + " " + str(freq) + "\n")

def add_vocabs_to_set(vocabs, vocab_set):
    for vocab in vocabs:
        if vocab in vocab_set:
            vocab_set[vocab] += 1
        else:
            vocab_set[vocab] = 1

def train_pair_from_dataset_pg(dataset_path, src_path, tgt_path):
    dataset = load_dataset(dataset_path)
    longest = 0
    length = 0
    with open(src_path, 'wt') as src_file, open(tgt_path, 'wt') as tgt_file:
        for item in dataset:
            abstract = item['abstract'].lower()
            related_work = sub("@cite_\S*", "@cite", item['related_work'].lower())
            ref_abstracts = item['ref_abstract']

            summary = ' '.join(word_tokenize(related_work))
            article = word_tokenize(abstract)
            for _, (_, reference) in enumerate(ref_abstracts.items()):
                article += word_tokenize(reference['abstract'])

            src_file.write(article + '\n')
            tgt_file.write(summary + '\n') 

    print(length/len(dataset))

def gen_train_files_BART(dataset_path, src_path, tgt_path):
    dataset = load_dataset(dataset_path)

    with open(src_path, 'wt') as src_file, open(tgt_path, 'wt') as tgt_file:
        for item in dataset:
            abstract = item['abstract']
            related_work = sub("@cite_\S*", "@cite", item['related_work'])
            ref_abstracts = item['ref_abstract']

            summary = related_work
            article = abstract

            for _, (_, reference) in enumerate(ref_abstracts.items()):
                if reference['abstract'] != '':
                    article += ' SEP ' + reference['abstract']

            src_file.write(article + '\n')
            tgt_file.write(summary + '\n')

if __name__ == "__main__":
    gen_train_files_BART("./dataset/Multi-XScience/val.json.gz", "val.source", "val.target")


