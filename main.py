import torch
from torch.utils.data import DataLoader
from embedding import BERTEmbedding
from dataset import BERTDataset
from transformers import BertTokenizer
from bert_tokenizer import get_bert_tokenizer
from bert import BERT
from pretraining import BERTLM, BERTTrainer

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
    MAX_LEN = 64

    try:
        tokenizer = BertTokenizer.from_pretrained("./bert-it-1/bert-it-vocab.txt", local_files_only = True)
    except FileNotFoundError:
        get_bert_tokenizer(data)
        tokenizer = BertTokenizer.from_pretrained("./bert-it-1/bert-it-vocab.txt", local_files_only = True)
    
    vocab_size = tokenizer.vocab_size
    embedding_size = 64

    train_data = BERTDataset(
        data, context_window = MAX_LEN, tokenizer = tokenizer
    )

    train_loader = DataLoader(
    train_data, batch_size = 32, shuffle = True, pin_memory = True)

    bert_model = BERT(
    vocab_size = len(tokenizer.vocab),
    d_model = 64,
    n_layers = 1,
    n_heads = 2,
    dropout = 0.1)

    bert_lm = BERTLM(bert_model, len(tokenizer.vocab))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_trainer = BERTTrainer(bert_lm , train_loader, device = device)
    epochs = 1


    for epoch in range(epochs):
        bert_trainer.train(epoch)

