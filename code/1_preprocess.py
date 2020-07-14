import config, utils_common
import torch
import numpy as np
np.random.seed(42)
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statistics import mean

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

def get_embedding(input_string, tokenizer, model):
    input_ids = torch.tensor([tokenizer.encode(input_string, add_special_tokens = True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0].numpy()  # Models outputs are now tuples
        assert last_hidden_states.shape == (1, 3, 768)
        last_hidden_states = last_hidden_states[:, 0, :]
        # last_hidden_states = np.mean(last_hidden_states, axis = 1)
        last_hidden_states = last_hidden_states.flatten()
        # print(input_string, input_ids, last_hidden_states.shape)
        return last_hidden_states

def save_word_to_embedding_pickle(word_to_aoa, output_path):

    bert_vocab = tokenizer.get_vocab()
    print(f"{len(bert_vocab)} in bert vocab, {len(word_to_aoa)} in aoa or abstractness")

    word_to_embedding = {}

    for word in tqdm(list(word_to_aoa.keys())):
        if word in bert_vocab:
            embedding = get_embedding(word, tokenizer, model)
            word_to_embedding[word] = embedding 
    
    utils_common.save_pickle(word_to_embedding, output_path)
    print(f"{len(word_to_embedding)} words saved")

if __name__ == "__main__":
    # word_to_aoa = utils_common.load_pickle(config.aoa_dict_path)
    # save_word_to_embedding_pickle(word_to_aoa, config.aoa_embedding_path)
    # word_to_embedding = utils_common.load_pickle(config.aoa_embedding_path)
    word_to_abstractness = utils_common.load_pickle(config.abstractness_dict_path)
    save_word_to_embedding_pickle(word_to_abstractness, config.abstractness_embedding_path)
    word_to_embedding = utils_common.load_pickle(config.abstractness_embedding_path)