import os
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
import tqdm

MAX_LEN = 64


def get_bert_tokenizer(pairs):
    os.makedirs("./data", exist_ok = True)
    text_data = []
    file_count = 0

    for sample in tqdm.tqdm(x[0] for x in pairs):
        text_data.append(sample)
        
        if len(text_data) == 10000:
            with open(f'./data/text_{file_count}.txt', "w", encoding = "utf-8") as fp:
                fp.write("\n".join(text_data))
            text_data = []
            file_count += 1
    paths = [str(x) for x in Path("./data").glob("**/*txt")]

    tokenizer = BertWordPieceTokenizer(
        clean_text = True,
        handle_chinese_chars = False,
        strip_accents = False,
        lowercase = True
    )

    tokenizer.train(
        files = paths,
        vocab_size = 30_000,
        min_frequency = 5,
        limit_alphabet = 1000,
        wordpieces_prefix = "##",
        special_tokens = ["[PAD]", '[CLS]', "[SEP]", "[MASK]", "[UNK]"]
    )

    os.makedirs("./bert-it-1", exist_ok = True)
    tokenizer.save_model("./bert-it-1", "bert-it")
    
    