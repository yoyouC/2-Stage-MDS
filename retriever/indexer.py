import sys
import lucene
 
from java.io import File
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, TextField, Field, StringField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory, RAMDirectory
from org.apache.lucene.util import Version

from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader
from org.apache.lucene.queryparser.classic import QueryParser

from data_loader import load_dataset

def addDoc(writer, abstract, mid, date, cite, rec_mid):
    doc = Document()
    doc.add(TextField("abstract", abstract.lower(), Field.Store.YES))
    doc.add(StringField("mid", mid.lower(), Field.Store.YES))
    doc.add(StringField("date", date.lower(), Field.Store.YES))
    doc.add(StringField("cite", cite.lower(), Field.Store.YES))
    doc.add(StringField("rec_mid", rec_mid.lower(), Field.Store.YES))
    writer.addDocument(doc)

if __name__ == "__main__":
    lucene.initVM()

    analyzer = StandardAnalyzer()
    path = Paths.get("retriever/runtime/index_all_extra_plus/")
    directory = SimpleFSDirectory.open(path)
    config = IndexWriterConfig(analyzer)
    w = IndexWriter(directory, config)

    dataset = load_dataset("dataset/all_reformat_extra_plus")

    references = dataset['references']

    for reference in references:
        if 'date' in reference and 'rec_mid' in reference:
            addDoc(w, reference['abstract'] + ' ' + reference['paper_title'] + ' ' + reference['paper_title'], reference['mid'], reference['date'], str(reference['citation_count']), str(reference['rec_mid']))
        else:
            addDoc(w, reference['abstract'], reference['mid'], 'UNKNOWN', 'UNKNOWN', 'UNKNOWN')

    w.close()



