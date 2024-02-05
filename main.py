from torch.utils.data import DataLoader
from embedding import BERTEmbedding
from dataset import BERTDataset
from transformers import BertTokenizer
from bert_tokenizer import get_bert_tokenizer

def get_data():
    corpus_movie_conv = './datasets/movie_conversations.txt'
    corpus_movie_lines = './datasets/movie_lines.txt'
    with open(corpus_movie_conv, "r", encoding = "iso-8859-1") as c:
        conv = c.readlines()
    with open(corpus_movie_lines, "r", encoding = "iso-8859-1") as l:
        lines = l.readlines()
        
    lines_dic = {}
    for line in lines:
        objects = line.split(" +++$+++ ")
        lines_dic[objects[0]] = objects[-1]

    pairs = []
    for con in conv:
        ids = eval(con.split(" +++$+++ ")[-1])
        for i in range(len(ids) - 1):
            qa_pairs = []
            
            first = lines_dic[ids[i]].strip()
            second = lines_dic[ids[i+1]].strip()
            
            qa_pairs.append(first)
            qa_pairs.append(second)
            pairs.append(qa_pairs)

    return pairs

if __name__ == "__main__":
    context_window = 64

    data = get_data()

    try:
        tokenizer = BertTokenizer.from_pretrained("./bert-it-1/bert-it-vocab.txt", local_files_only = True)
    except FileNotFoundError:
        get_bert_tokenizer(data)
        tokenizer = BertTokenizer.from_pretrained("./bert-it-1/bert-it-vocab.txt", local_files_only = True)
    
    vocab_size = tokenizer.vocab_size
    embedding_size = 768

    dataset = BERTDataset(data, tokenizer, context_window)
    train_loader = DataLoader(
        dataset, batch_size=32, shuffle = True, pin_memory=True
    )

    embeddings = BERTEmbedding(vocab_size, embedding_size, context_window)
    sample = next(iter(train_loader))

    print(embeddings(sample["input"], sample["segment_label"]).shape)

    

