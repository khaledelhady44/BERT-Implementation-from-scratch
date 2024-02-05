import random
import torch
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    """
    Construct the dataset required for pretraining bert.
    """
    def __init__(self, data, tokenizer, context_window):

        # The tokenizer that we will use (word piece for bert).
        self.tokenizer = tokenizer  

        # Size of  input_tokens entered to the model.
        self.context_window = context_window 
        
        # Pretraining data in the shape of a list of pairs. 
        # Each pair contains two sentences
        self.data = data 

        # vocab_size
        self.vocab_size = len(self.tokenizer.vocab)

    def __len__(self):
        """Returns how many data pairs are in the dataset."""
        return len(self.data)
    
    def __getitem__(self, item):
        """Prepares an item to be processed by the model."""

        # First let's use get_pair to get 2 sentences form our dataset.
        # Second sentence could follow first sentence or could be random.
        # is_next tells us if two sentences follow each other.
        sentence1, sentence2, is_next = self.get_pair(item)

        # Modify both sentences by selecting random words to be masked.
        # Lable is what the model should predict for the sentence
        sentence1_modified, sentence1_label = self.modify_sentence(sentence1)
        sentence2_modified, sentence2_label = self.modify_sentence(sentence2)


        # some special tokens used by the model
        cls = self.tokenizer.vocab["[CLS]"]
        sep = self.tokenizer.vocab["[SEP]"]
        pad = self.tokenizer.vocab["[PAD]"]


        # first sentence with bert starts with cls then the modeifed versoin
        # after masking then sep token
        sentence1 = [cls] + sentence1_modified + [sep]

        # second sentence ends with sep token
        sentence2 = sentence2_modified + [sep]

        # the label for special tokens is 0 which is padding.
        sentence1_label = [pad] + sentence1_label + [pad] # cls_label + s1_label + sep label
        sentence2_label  = sentence2_label + [pad]

        # segment label tells the model whether this token follows
        # sentence1 or sentence2.
        # length needs to be context_window not larger.
        segment_label = ([1 for _ in range(len(sentence1))] + [2 for _ in range(len(sentence2))])[:self.context_window]

        # input to the model is the two sentence concatenated together.
        input_to_model = (sentence1 + sentence2)[:self.context_window]
        label = (sentence1_label + sentence2_label)[:self.context_window]

        # If length of both sentence is smaller that context_window
        # we need to add pad tokens to the end of input and label and segment_label.
        
        padding = [pad for _ in range(self.context_window - len(input_to_model))]

        # add padding to input and labels
        input_to_model.extend(padding)
        label.extend(padding)
        segment_label.extend(padding)

        output = {"input": input_to_model,
                  "label": label,
                  "segment_label": segment_label,
                  "is_next": is_next}

        return {key: torch.tensor(value) for key, value in output.items()}
    
    def modify_sentence(self, sentence):
        """
        tokenize the sentence and randomly mask words.

        Returns the tokenized sentence after random masking and also the correct label.
        """
        sentence_modified = []
        sentence_label = []
        mask = self.tokenizer.vocab["[MASK]"]

        for word in sentence.split():
            # this is the probability to choose a specific word.
            choose_prob = random.random()

            # tokenize the words and neglect the first and last token -> [CLS], [SEP]
            tokenized_word = self.tokenizer(word)["input_ids"][1:-1]

            # we will choos word with probability of 15%.
            if choose_prob < 0.15:
                
                # we will mask the word with probability 80%.
                # replace it with a random token with probability 10%.
                # leave it as it is with probability 10%.
                mask_prob = random.random()

                if mask_prob < 0.8:
                    for _ in range(len(tokenized_word)):
                        # add mask tokens
                        sentence_modified.append(mask)
                elif mask_prob < 0.9:
                    for _ in range(len(tokenized_word)):
                        # get random tokens
                        sentence_modified.append(random.randrange(self.vocab_size))
                else:
                    # leave the word as it is
                    sentence_modified.extend(tokenized_word)

                # since we choosed the word we will add it in the label.
                sentence_label.extend(tokenized_word)
            else:
                # we will not mask the word so let's add it as it is.
                sentence_label.extend(tokenized_word)

                # label is just zeros because we don't want the model to predict anything for these words.
                sentence_modified.extend([0] * len(tokenized_word))
            
            # make sure that label has same size as input
            assert len(sentence_modified) == len(sentence_modified)

            return sentence_modified, sentence_label
        
    def get_pair(self, item):
        """returns 2 sentences from the data."""

        # with probability 50% we will get 2 sentences next to each other.
        # in the other 50% of times we will get 2 random sentences.
        if random.random() > 0.5:
            return self.data[item][0], self.data[item][1], 1
        else:
            return self.data[item][0], self.data[random.randrange(len(self))][1], 0
