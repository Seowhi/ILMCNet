import torch.nn as nn
from net.crf import CRF
import numpy as np
from sklearn.metrics import f1_score, classification_report
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertLayerNorm
from transformers import T5Tokenizer, T5EncoderModel
import torch
import torch.nn as nn
import math
import config.args as args


class BiLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMLayer, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.fc(out)
        return out

class MultiCNNLayer(nn.Module):
    def __init__(self):
        super(MultiCNNLayer, self).__init__()
        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv1d(args.hidden_size, 1024, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Conv1d(1024, 512, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Conv1d(512, 256, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.elmo_feature_extractor(x)
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(args.hidden_size, args.num_tags)
        self.linear = nn.Linear(args.hidden_size, 1040)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(256, args.num_tags, kernel_size=7, padding=3)
        )
        self.relu = nn.ReLU()
        self.fnn1 = nn.Linear(256, 128)
        self.fnn2 = nn.Linear(128, 64)
        self.fnn3 = nn.Linear(64, args.num_tags)
        # self.fnn = nn.Linear(256, args.num_tags)

        # self.lstm = BiLSTMLayer(256, 256 // 2, 256)
        self.lstm = nn.LSTM(256, 256 // 2, bidirectional=True)
        self.cnnlayer = MultiCNNLayer()
        self.crf = CRF(args.num_tags)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=8)  #8->4
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)  #6 -> 4
        # self.embeddings = BertEmbeddings()
        self.embeddings = SeqEmbedding(args.vocab_count, args.encode_size, 0.1, args.device)    #d_model=args.encode_size


    def forward(self, tensor):
        tensor = tensor.to(args.device).float()
        # tensor = self.linear(tensor)
        transformer_out = self.transformer(tensor)   # torch.Size([2, 880, 2048])
        # print("transformer_out shape:", transformer_out.shape)
        # exit()

        cnnlayer_out = self.cnnlayer(transformer_out.permute(0, 2, 1))  # torch.Size([2, 256, 880])
        # print("cnnlayer_out shape:", cnnlayer_out.shape)

        lstm_out, _ = self.lstm(cnnlayer_out.permute(0, 2, 1))  # torch.Size([2, 880, 256])
        # print("lstm_out shape:", lstm_out.shape)

        # output = self.conv(lstm_out.permute(0, 2, 1))  # torch.Size([2, 3, 700])
        # return output.permute(0, 2, 1)

        lstm_out = self.relu(lstm_out)
        fnn_output_1 = self.fnn1(lstm_out)
        fnn_output_2 = self.fnn2(fnn_output_1)
        fnn_output = self.fnn3(fnn_output_2)
        # fnn_output = self.fnn(lstm_out)


        return fnn_output

    def loss_fn(self, encode, output_mask, tags):
        loss = self.crf.negative_log_loss(encode, output_mask, tags)
        return loss

    def predict(self, encode, output_mask):
        predicts = self.crf.get_batch_best_path(encode, output_mask)
        predicts = predicts.view(1, -1).squeeze()
        predicts = predicts[predicts != -1]
        return predicts

    def acc_f1(self, y_pred, y_true):

        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        f1 = f1_score(y_true, y_pred, average="macro")
        correct = np.sum((y_true == y_pred).astype(int))
        acc = correct / y_pred.shape[0]
        return acc, f1

    def class_report(self, y_pred, y_true):
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        classify_report = classification_report(y_true, y_pred)
        print('\n\nclassify_report:\n', classify_report)



    # BertEmbeddings
    # def get_embeddings(self, input_ids, token_type_ids):
    #     # embedding_output = self.embeddings(input_ids, token_type_ids)
    #     embedding_output = self.embeddings(input_ids)
    #     return embedding_output

    #TransformerEmbedding
    def get_embeddings(self, input_ids, token_type_ids, max_len):
        embedding_output = self.embeddings(input_ids, max_len)
        return embedding_output

    # def t5_code(self, t5_encode):
    #     lstm_out, _ = self.lstm(t5_encode)  # torch.Size([2, 700, 1024])
    #     # print(bert_encode.shape)
    #     # exit()
    #
    #     # output = self.classifier(lstm_out)  # torch.Size([2, 700, 3])
    #
    #     #------------把最后分类的线形层变成CNN层-------------------
    #     lstm_out = lstm_out.permute(0, 2, 1)
    #     output = self.conv(lstm_out)  # torch.Size([2, 3, 700])
    #
    #     return output.permute(0, 2, 1)


# hidden_size=1024,
# num_hidden_layers=12,
# num_attention_heads=12,
# intermediate_size=4096,
# hidden_act="gelu",
# hidden_dropout_prob=0.0,
# attention_probs_dropout_prob=0.0,
# max_position_embeddings=40000,
# type_vocab_size=2,
# initializer_range=0.02,
# vocab_size=30

# class BertEmbeddings(nn.Module):
#     """Construct the embeddings from word, position and token_type embeddings.
#     """
#
#     def __init__(self):
#         super(BertEmbeddings, self).__init__()
#         self.word_embeddings = nn.Embedding(args.vocab_count, 1024, padding_idx=0)  # 5406是vocab.txt写入的个数
#         self.position_embeddings = nn.Embedding(40000, 1024)
#         self.token_type_embeddings = nn.Embedding(2, 1024)
#
#         # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
#         # any TensorFlow checkpoint file
#         self.LayerNorm = BertLayerNorm(1024, eps=1e-12)
#         self.dropout = nn.Dropout(0.0)
#
#     def forward(self, input_ids, token_type_ids=None):
#         seq_length = input_ids.size(1)
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
#         position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
#         if token_type_ids is None:
#             token_type_ids = torch.zeros_like(input_ids)
#
#         words_embeddings = self.word_embeddings(input_ids)
#         position_embeddings = self.position_embeddings(position_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)
#
#         embeddings = words_embeddings + position_embeddings + token_type_embeddings
#         # embeddings = words_embeddings + position_embeddings
#         # print("embeddings shape:", embeddings.shape)
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len=5000, device='cpu'):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x, max_len):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class SeqEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, drop_prob, device):
        """
        class for word embedding that included positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(SeqEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len=5000, device=device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x, max_len):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x, max_len)
        return self.drop_out(tok_emb + pos_emb)
