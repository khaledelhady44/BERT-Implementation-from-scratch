from dataset import BERTDataset

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
    data = get_data()