import torch
from collections import Counter
import pickle, sys, json, io
from os.path import expanduser
HOME = expanduser("~")
from tqdm import tqdm
import numpy as np

class Vocabulary:
    """Vocabulary class for mapping between words and ids"""
    def __init__(self, word2idx=None, idx2word=None, field=None, corpus=None, max_words=50000, min_frequency=5,
                 start_end_tokens=True):
        if corpus is None:
            self.word2idx = word2idx
            self.idx2word = idx2word
            self.start_end_tokens = False
            self.size = len(word2idx)
        else:
            self.word2idx = dict()
            self.idx2word = dict()
            self.vocabulary = None
            # most common words
            self.max_words = max_words
            # least common words
            self.min_frequency = min_frequency
            self.start_end_tokens = start_end_tokens
            self._build_vocabulary(corpus, field)
            print("Finish build vocabulary")
            self._build_word_index()
            print("Finish build word dictionary")
            self.size = len(self.word2idx)

    def _build_vocabulary(self, corpus, field):
        vocabulary = Counter(word for sent in corpus for word in sent)
        field_vocab = Counter(word for sent in field for word in sent)
        # print(field_vocab)
        if self.max_words:
            vocabulary = {word: freq for word,
                          freq in vocabulary.most_common(self.max_words)}
        if self.min_frequency:
            vocabulary = {word: freq for word, freq in vocabulary.items()
                          if freq >= self.min_frequency}
        self.vocabulary = Counter(vocabulary)
        self.vocabulary.update(field_vocab)
        # self.size = len(self.vocabulary) + 2  # padding and unk tokens
        # if self.start_end_tokens:
        #     self.size += 2

    def _build_word_index(self):
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = 1

        if self.start_end_tokens:
            self.word2idx['<EOS>'] = 2
            self.word2idx['<SOS>'] = 3

        offset = len(self.word2idx)
        for idx, word in enumerate(self.vocabulary):
            self.word2idx[word] = idx + offset
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def add_start_end(self, vector):
        vector.append(self.word2idx['<EOS>'])
        return [self.word2idx['<SOS>']] + vector

    def vectorize_field(self, vector):
        _o_field = []
        for word in vector:
            _o_field.append(self.word2idx[word])
        return _o_field

    def vectorize_source(self, vector, table):
        source_oov = {}  # {oov: id start from 0}
        _o_source, _source = [], []
        cnt = 0
        oov_freq = 0
        for word in vector:
            try:
                _o_source.append(self.word2idx[word])
                _source.append(self.word2idx[word])
            except KeyError:
                oov_freq += 1
                if word not in source_oov:
                    _o_source.append(cnt + self.size)
                    source_oov[word] = cnt
                    cnt += 1
                else:
                    _o_source.append(source_oov[word] + self.size)
                # NOTE: use field type embedding for OOV field values
                _source.append(self.word2idx[table[word]])
        return _o_source, source_oov, _source, oov_freq

    def vectorize_target(self, vector, source_oov, table):
        _o_target, _target = [], []
        oov_freq = 0
        for word in vector:
            try:
                _o_target.append(self.word2idx[word])
                _target.append(self.word2idx[word])
            except KeyError:
                oov_freq += 1
                # NOTE: use UNK for text words not in source OOVs
                if word not in source_oov:
                    _o_target.append(self.word2idx['<UNK>'])
                    _target.append(self.word2idx['<UNK>'])
                else:
                    # NOTE: use source_oov idx for OOV text words but appear in source OOVs
                    _o_target.append(source_oov[word] + self.size)
                    _target.append(self.word2idx[table[word]])
        return self.add_start_end(_o_target), self.add_start_end(_target), oov_freq


class Table2text_seq:
    def __init__(self, data_src, type=0, batch_size=128, USE_CUDA=torch.cuda.is_available(),
                 train_mode=False, dec_type='pg'):
        prefix = "{}/table2text_nlg/data/dkb/".format(HOME)
        self.type = type
        self.dec_type =dec_type
        self.vocab = None
        self.text_len = 0
        self.max_p = 0
        self.data_src = data_src
        self.batch_size = batch_size
        self.USE_CUDA = USE_CUDA
        self.device = torch.device("cuda" if USE_CUDA else "cpu")

        self.oov_cnt_src = 0
        self.oov_cnt_tgt = 0
        self.total_cnt_src = 0
        self.total_cnt_tgt = 0

        # ----------------------- file names ------------------------- #
        if data_src == 'train':
            if self.type == 0:
                path = "{}train_P.pkl".format(prefix)
            else:
                path = "{}train_A.pkl".format(prefix)
        elif data_src == 'valid':
            if self.type == 0:
                path = "{}valid_P.pkl".format(prefix)
            else:
                path = "{}valid_A.pkl".format(prefix)
        elif data_src == 'test':
            if self.type == 0:
                path = "{}test_P.pkl".format(prefix)
            else:
                path = "{}test_A.pkl".format(prefix)
        else:
            raise ValueError("Only train, valid, test data_src are supported")

        # ----------------------- load triples and build vocabulary ------------------------- #
        if data_src == 'train' and not train_mode:
            self.load_data_light(path)
        else:
            self.data = self.load_data(path)
            self.len = len(self.data)
            # ----------------------- word to ids ------------------------- #
            self.corpus = self.batchfy()

        print("vocab size = {}".format(self.vocab.size))

    def load_data_light(self, path):
        print("Loading data $LIGHT$ from {}".format(path))
        prefix = "{}/table2text_nlg/describe_kb/outputs".format(HOME)
        if self.type == 0:
            vocab_path_pkl = "{}/dkb_vocab.pkl".format(prefix)
        else:
            vocab_path_pkl = "{}/dkb_vocab_D.pkl".format(prefix)
        print("loading vocab ...")
        with open(vocab_path_pkl, 'rb') as fin:
            data = pickle.load(fin)
        self.vocab = Vocabulary(word2idx=data["word2idx"], idx2word=data["idx2word"])

        print("Loading data from {}".format(path))
        # (qkey, qitem, index)
        with open(path, 'rb') as fin:
            data = pickle.load(fin)
        old_sources = data["source"]
        print("{} samples to be processed".format(len(old_sources)))
        for idx, old_source in enumerate(tqdm(old_sources)):
            p_for = []
            for key, value, pos in old_source:
                p_for.append(pos)
            curr_p_max = max(p_for) + 1
            if self.max_p < curr_p_max:
                self.max_p = curr_p_max

    def load_data(self, path):
        # TODO: tokenize field and texts
        prefix = "{}/table2text_nlg/describe_kb/outputs".format(HOME)
        print("Loading data from {}".format(path))
        # (qkey, qitem, index)
        with open(path, 'rb') as fin:
            data = pickle.load(fin)
        old_sources = data["source"]
        old_targets = data["target"]
        total = []
        samples = []
        total_field = []

        print("{} samples to be processed".format(len(old_sources)))
        for idx, old_source in enumerate(tqdm(old_sources)):
            # print("old_source: {}".format(old_source))
            source = []
            field = []
            table = {}
            p_for = []
            p_bck = []
            target = old_targets[idx]
            target = [x.lower() for x in target]  # NOTE: changed to lowercase strings
            if len(target) > self.text_len:
                self.text_len = len(target) + 2
            for key, value, index in old_source:
                value = value.lower()  # NOTE: changed to lowercase strings
                tag = '<'+key+'>'  # NOTE: change key into special tokens
                source.append(value)
                field.append(tag)
                p_for.append(index)
                if value not in table:
                    table[value] = tag
            curr_p_max = max(p_for) + 1
            for p in p_for:
                p_bck.append(curr_p_max - p)
            if self.max_p < curr_p_max:
                self.max_p = curr_p_max
            # print("source: {}".format(source))
            # print("field: {}".format(field))
            total.append(source + target)
            total_field.append(field)
            samples.append([source, target, field, p_for, p_bck, table])

        '''
            torch.nn.utils.rnn.pack_padded_sequence requires the sequence lengths sorted in decreasing order
        '''
        print("sorting samples ...")
        self.sort_indices = np.argsort([len(x[0]) for x in samples]).tolist()
        self.sort_indices.reverse()
        samples = np.array(samples)[self.sort_indices].tolist()

        if self.type == 0:
            vocab_path_pkl = "{}/dkb_vocab.pkl".format(prefix)
            vocab_path_js = "{}/dkb_vocab.json".format(prefix)
        else:
            vocab_path_pkl = "{}/dkb_vocab_D.pkl".format(prefix)
            vocab_path_js = "{}/dkb_vocab_D.json".format(prefix)
        if self.data_src == 'train':
            if self.type == 0:
                self.vocab = Vocabulary(corpus=total, field=total_field)
            else:
                self.vocab = Vocabulary(corpus=total, field=total_field, min_frequency=0)
            data = {
                "idx2word": self.vocab.idx2word,
                "word2idx": self.vocab.word2idx
            }
            with open(vocab_path_pkl, 'wb') as fout:
                pickle.dump(data, fout)
            with io.open(vocab_path_js, 'w', encoding='utf-8') as fout:
                json.dump(data, fout, sort_keys=True, indent=4)
        else:
            with open(vocab_path_pkl, 'rb') as fin:
                data = pickle.load(fin)
            self.vocab = Vocabulary(word2idx=data["word2idx"], idx2word=data["idx2word"])
        return samples

    def batchfy(self):
        print("Constructing Batches ...")
        samples = [self.data[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        corpus = []
        for sample in tqdm(samples):
            corpus.append(self.vectorize(sample))
        print("oov_cnt_src: {0} ({1:.3f}%)".format(self.oov_cnt_src, 100.0*self.oov_cnt_src/self.total_cnt_src))
        print("oov_cnt_tgt: {0} ({1:.3f}%)".format(self.oov_cnt_tgt, 100.0*self.oov_cnt_tgt/self.total_cnt_tgt))
        print("total_cnt_src: {}".format(self.total_cnt_src))
        print("total_cnt_tgt: {}".format(self.total_cnt_tgt))
        return corpus

    def pad_vector(self, vector, maxlen):
        padding = maxlen - len(vector)
        vector.extend([0] * padding)
        return vector

    def vectorize(self, sample):
        """
            batch_s         --> tensor batch of table with ids
            batch_o_s       --> tensor batch of table with ids and <unk> replaced by temp OOV ids
            batch_t         --> tensor batch of text with ids
            batch_o_t       --> tensor batch of target and <unk> replaced by temp OOV ids
            batch_f         --> tensor batch of field with ids(might not exist)
            batch_pf        --> tensor batch of forward position
            batch_pb        --> tensor batch of backward position
            batch_o_f       --> tensor batch of field and used wordid
            max_article_oov --> max number of OOV tokens in article batch
        """

        # print(len(sample))
        batch_o_s, batch_o_t, batch_f, batch_t, batch_s, batch_pf, batch_pb = [], [], [], [], [], [], []
        source_len, target_len, w2fs = [], [], []
        list_oovs = []
        targets = []
        sources = []
        fields = []
        max_source_oov = 0
        for data in sample:
            source = data[0]
            target = data[1]
            field = data[2]
            p_for = data[3]
            p_bck = data[4]
            table = data[5]
            source_len.append(len(source))
            target_len.append(len(target) + 2)

            # ----------------------- word to ids ------------------------- #
            _fields = self.vocab.vectorize_field(field)
            _o_source, source_oov, _source, oov_freq_src = self.vocab.vectorize_source(source, table)
            _o_target, _target, oov_freq_tgt = self.vocab.vectorize_target(target, source_oov, table)

            self.oov_cnt_src += oov_freq_src
            self.oov_cnt_tgt += oov_freq_tgt
            self.total_cnt_src += len(_source)
            self.total_cnt_tgt += len(_target)

            source_oov = source_oov.items()
            if max_source_oov < len(source_oov):
                max_source_oov = len(source_oov)
            if self.data_src != 'train':
                idx2word_oov = {idx: word for word, idx in source_oov}
                w2f = {(idx+self.vocab.size): self.vocab.word2idx[table[word]] for word, idx in source_oov}
                w2fs.append(w2f)
                list_oovs.append(idx2word_oov)
                targets.append(target)  # tokens
                sources.append(source)  # tokens
                fields.append(field)    # tokens

            batch_s.append(_source)
            batch_o_s.append(_o_source)
            batch_f.append(_fields)
            batch_pf.append(p_for)
            batch_pb.append(p_bck)

            batch_t.append(_target)
            batch_o_t.append(_o_target)

        batch_s = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_s]
        batch_f = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_f]
        batch_pf = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_pf]
        batch_pb = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_pb]
        batch_o_s = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_o_s]

        batch_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_t]
        batch_o_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_o_t]

        batch_s = torch.stack(batch_s, dim=0)
        batch_f = torch.stack(batch_f, dim=0)
        batch_pf = torch.stack(batch_pf, dim=0)
        batch_pb = torch.stack(batch_pb, dim=0)
        batch_o_s = torch.stack(batch_o_s, dim=0)

        batch_t = torch.stack(batch_t, dim=0)
        batch_o_t = torch.stack(batch_o_t, dim=0)

        if self.data_src != 'train':
            targets= [i[:max(target_len)-2] for i in targets]
            sources= [i[:max(source_len)] for i in sources]
            fields = [i[:max(source_len)] for i in fields]
            return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, targets, sources, fields, list_oovs, source_len, \
                max_source_oov, w2fs
        else:
            return batch_s, batch_o_s, batch_t, batch_o_t, batch_f, batch_pf, batch_pb, source_len, max_source_oov

    def get_batch(self, index):
        if self.data_src == 'train':
            batch_s, batch_o_s, batch_t, batch_o_t, batch_f, batch_pf, batch_pb, source_len, max_source_oov \
                = self.corpus[index]
            batch_s = batch_s.to(self.device)
            batch_o_s = batch_o_s.to(self.device)
            batch_t = batch_t.to(self.device)
            batch_o_t = batch_o_t.to(self.device)
            batch_f = batch_f.to(self.device)
            batch_pf = batch_pf.to(self.device)
            batch_pb = batch_pb.to(self.device)
            return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, batch_t, batch_o_t, source_len, max_source_oov
        else:
            batch_s, batch_o_s, batch_f, batch_pf, batch_pb, targets, sources, fields, list_oovs, source_len, \
                max_source_oov, w2fs = self.corpus[index]
            batch_s = batch_s.to(self.device)
            batch_o_s = batch_o_s.to(self.device)
            batch_f = batch_f.to(self.device)
            batch_pf = batch_pf.to(self.device)
            batch_pb = batch_pb.to(self.device)
            return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, sources, targets, fields, list_oovs, source_len, \
                max_source_oov, w2fs

