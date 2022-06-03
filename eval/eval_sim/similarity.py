from random import randint

import numpy as np
import torch

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def test(sent1, sent2):
    
    # Load model
    from models import InferSent
    model_version = 1
    MODEL_PATH = "encoder/infersent%s.pkl" % model_version
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Keep it on CPU or put it on GPU
    use_cuda = False
    model = model.cuda() if use_cuda else model

    # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
    W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
    model.set_w2v_path(W2V_PATH)

    # Load embeddings of K most frequent words
    model.build_vocab_k_words(K=100000)

    return cosine(model.encode([sent1])[0], model.encode([sent2])[0])

def main(src_path, hyp_path):
    # Load model
    from models import InferSent
    model_version = 1
    MODEL_PATH = "encoder/infersent%s.pkl" % model_version
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Keep it on CPU or put it on GPU
    use_cuda = False
    model = model.cuda() if use_cuda else model

    # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
    W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
    model.set_w2v_path(W2V_PATH)

    # Load embeddings of K most frequent words
    model.build_vocab_k_words(K=100000)


    fs = open(src_path, 'rt')
    fh = open(hyp_path, 'rt')

    tot_sim = 0
    num = 0

    while 1:
        line1 = fs.readline()
        line2 = fh.readline()
        
        if line1 == '' and line2 == '':
            break
        
        tot_sim += cosine(model.encode([line1])[0], model.encode([line2])[0])
        num += 1

    mean_sim = tot_sim/num

    print(f"Mean Sentence Similarity: {mean_sim} for {num} lines")

if __name__ == '__main__':
    print(test("this", "these"))

    