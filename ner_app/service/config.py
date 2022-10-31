TOTAL_DIR = ''

ORIGIN_DIR = TOTAL_DIR+'/input/origin/'
ANNOTATION_DIR = TOTAL_DIR+'/output/annotation/'

TRAIN_SAMPLE_PATH = TOTAL_DIR+'/output/train_sample.txt'
TEST_SAMPLE_PATH = TOTAL_DIR+'/output/test_sample.txt'

VOCAB_PATH = TOTAL_DIR+'ner_app/service/ner_model_torch/vocab.txt'
LABEL_PATH = TOTAL_DIR+'ner_app/service/ner_model_torch/label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

import torch
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


VOCAB_SIZE = 3000
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
TARGET_SIZE = 31
LR = 1e-4
EPOCH = 3

MODEL_DIR = TOTAL_DIR+'ner_app/service/ner_model_torch/'