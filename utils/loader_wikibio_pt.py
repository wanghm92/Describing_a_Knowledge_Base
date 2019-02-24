import torch
from collections import Counter
import pickle, sys, json, io, copy
from os.path import expanduser
HOME = expanduser("~")
from tqdm import tqdm

class Vocabulary:
    """Vocabulary class for mapping between words and ids"""
    def __init__(self, word2idx=None, idx2word=None, field2idx=None, idx2field=None, field=None, corpus=None,
                 max_words=20000, min_frequency=5, start_end_tokens=True, min_frequency_field=100, dec_type='pg'):
        self.dec_type =dec_type
        if corpus is None:
            self.word2idx = word2idx
            self.idx2word = idx2word
            self.field2idx = field2idx
            self.idx2field = idx2field
            self.start_end_tokens = False
            self.size = len(word2idx)
            self.field_vocab_size = len(field2idx)
            # self.field_vocab_size += 2  # TODO: remove this legacy
        else:
            self.word2idx = dict()
            self.idx2word = dict()
            self.size = 0
            self.vocabulary = None

            self.field2idx = dict()
            self.idx2field = dict()
            self.field_vocab_size = 0
            self.field_vocab = None

            # most common words
            self.max_words = max_words
            # least common words
            self.min_frequency = min_frequency
            self.min_frequency_field = min_frequency_field
            self.start_end_tokens = start_end_tokens
            self._build_vocabulary(corpus, field)
            print("Finish build vocabulary")
            self._build_word_index()
            print("Finish build word dictionary")
            self.size = len(self.word2idx)
            self.field_vocab_size = len(self.field2idx)

    def _build_vocabulary(self, corpus, field):
        vocabulary = Counter(word for sent in corpus for word in sent)
        field_vocab = Counter(word for sent in field for word in sent)
        if self.max_words:
            vocabulary = {word: freq for word,
                          freq in vocabulary.most_common(self.max_words)}
        if self.min_frequency:
            vocabulary = {word: freq for word, freq in vocabulary.items()
                          if freq >= self.min_frequency}
        if self.min_frequency_field:
            field_vocab = {word: freq for word, freq in field_vocab.items()
                           if freq >= self.min_frequency_field}

        self.vocabulary = Counter(vocabulary)
        self.field_vocab = Counter(field_vocab)
        # print(self.field_vocab)


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

        self.field2idx['<PAD>'] = 0
        self.field2idx['<UNK>'] = 1
        if self.dec_type == 'pt':
            self.field2idx['<EOS>'] = 2
            self.field2idx['<SOS>'] = 3

        offset = len(self.field2idx)
        for idx, fd in enumerate(self.field_vocab):
            self.field2idx[fd] = idx + offset
        self.idx2field = {idx: fd for fd, idx in self.field2idx.items()}

    def add_start_end(self, vector):
        vector.append(self.word2idx['<EOS>'])
        return [self.word2idx['<SOS>']] + vector

    def vectorize_field(self, vector, is_target=False):
        _o_field = []
        for fd in vector:
            _o_field.append(self.field2idx.get(fd, self.field2idx['<UNK>']))

        if self.dec_type == 'pt':
            _o_field.append(self.field2idx['<EOS>'])
            if is_target:
                _o_field = [self.field2idx['<SOS>']] + _o_field
        return _o_field

    def vectorize_source(self, vector, table):
        source_oov = {}
        _o_source, _source = [], []
        cnt = 0
        oov_freq = 0
        for word in vector:
            try:
                _o_source.append(self.word2idx[word])
                _source.append(self.word2idx[word])
            except KeyError:
                oov_freq += 1
                if self.dec_type == 'pg':
                    if word not in source_oov:
                        _o_source.append(cnt + self.size)
                        source_oov[word] = cnt
                        cnt += 1
                    else:
                        _o_source.append(source_oov[word] + self.size)
                else:
                    _o_source.append(self.word2idx['<UNK>'])
                _source.append(self.word2idx['<UNK>'])
                # use field type embedding for OOV values
                # _source.append(self.word2idx.get(table[word], self.word2idx['<UNK>']))
        if self.dec_type == 'pt':
            _o_source.append(self.word2idx['<EOS>'])
            _source.append(self.word2idx['<EOS>'])
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
                _o_target.append(self.word2idx['<UNK>'])
                _target.append(self.word2idx['<UNK>'])
                # # NOTE: use UNK for text words not in source OOVs
                # if word not in source_oov:
                #     _o_target.append(self.word2idx['<UNK>'])
                #     _target.append(self.word2idx['<UNK>'])
                # else:
                #     # NOTE: use source_oov idx for OOV text words but appear in source OOVs
                #     _o_target.append(source_oov[word] + self.size)
                #     _target.append(self.word2idx['<UNK>'])
                #     # _target.append(self.word2idx.get(table[word], self.word2idx['<UNK>']))
        return self.add_start_end(_o_target), self.add_start_end(_target), oov_freq


class Table2text_seq:
    def __init__(self, data_src, type=0, batch_size=128, USE_CUDA=torch.cuda.is_available(),
                 train_mode=0, dec_type='pg'):
        # TODO: change the path
        prefix = "{}/table2text_nlg/data/dkb/wikibio_pt_data/".format(HOME)
        assert type == 2
        self.vocab = None
        self.dec_type = dec_type
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
            path = "{}train_P.pkl".format(prefix)
        elif data_src == 'valid':
            path = "{}valid_P.pkl".format(prefix)
        elif data_src == 'test':
            path = "{}test_P.pkl".format(prefix)
        else:
            raise ValueError("Only train, valid, test data_srcs are supported")

        # ----------------------- load triples and build vocabulary ------------------------- #
        if data_src == 'train' and (train_mode != 0 and train_mode != 1):  # training(0) and resume training(1)
            self.data = self.load_data_light(path)
        else:
            self.data = self.load_data(path)
            self.len = len(self.data)
            # ----------------------- word to ids ------------------------- #
            self.corpus = self.batchfy()

        print("vocab size = {}".format(self.vocab.size))

    def load_data_light(self, path):
        print("Loading data $LIGHT$ from {}".format(path))
        prefix = "{}/table2text_nlg/describe_kb/outputs".format(HOME)
        vocab_path_pkl = "{}/wikibio_vocab_pt.pkl".format(prefix)
        print("loading vocab ... from {}".format(vocab_path_pkl))
        with open(vocab_path_pkl, 'rb') as fin:
            data = pickle.load(fin)
        self.vocab = Vocabulary(word2idx=data["word2idx"], idx2word=data["idx2word"],
                                field2idx=data["field2idx"], idx2field=data["idx2field"],
                                dec_type=self.dec_type)

        print("Loading data $LIGHT$ from {}".format(path))
        # (qkey, qitem, index)
        with open(path, 'rb') as fin:
            data = pickle.load(fin)
        old_sources = data["source"]
        print("{} samples to be processed".format(len(old_sources)))
        for idx, old_source in enumerate(tqdm(old_sources)):
            p_for = []
            for key, value, pos, rpos in old_source:
                p_for.append(pos)
            curr_p_max = max(p_for) + 1
            if self.max_p < curr_p_max:
                self.max_p = curr_p_max


    def load_data(self, path):
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
            table = {}

            value_s = []
            field_s = []
            p_for_s = []
            p_bck_s = []
            for key, value, pos, rpos in old_source:
                value = value.lower()  # NOTE: changed to lowercase strings
                tag = '<'+key+'>'  # NOTE: change key into special tokens
                value_s.append(value)
                field_s.append(tag)
                p_for_s.append(pos)
                p_bck_s.append(rpos)

                if value not in table:
                    table[value] = tag

            old_target = old_targets[idx]
            temp = copy.deepcopy(old_target)
            if len(list(temp)) > self.text_len:
                self.text_len = len(list(temp)) + 2

            value_t = []
            field_t = []
            p_for_t = []
            p_bck_t = []
            lab_t = []
            for key, value, pos, rpos, lab in old_target:
                value = value.lower()  # NOTE: changed to lowercase strings
                tag = '<'+key+'>'  # NOTE: change key into special tokens
                value_t.append(value)
                field_t.append(tag)
                p_for_t.append(pos)
                p_bck_t.append(rpos)
                lab_t.append(lab)  # int

            # print("value_s: {}".format(value_s))
            # print("field_s: {}".format(field_s))
            # print("p_for_s: {}".format(p_for_s))
            # print("p_for_s: {}".format(p_bck_s))
            # print("value_t: {}".format(value_t))
            # print("field_t: {}".format(field_t))
            # print("p_for_t: {}".format(p_for_t))
            # print("p_for_t: {}".format(p_bck_t))
            # print("lab_t: {}".format(lab_t))

            curr_p_max = max(p_for_s) + 1

            if self.max_p < curr_p_max:
                self.max_p = curr_p_max
            # print("source: {}".format(source))
            # print("field: {}".format(field))
            target = (value_t, field_t, p_for_t, p_bck_t, lab_t)
            total.append(value_s + value_t)
            total_field.append(field_s + field_t)
            samples.append([value_s, target, field_s, p_for_s, p_bck_s, table])

        # TODO: reverse batches ???
        '''
            torch.nn.utils.rnn.pack_padded_sequence requires the sequence lengths sorted in decreasing order
        '''
        print("sorting samples ...")
        samples.sort(key=lambda x: len(x[0]), reverse=True)

        vocab_path_pkl = "{}/wikibio_vocab_pt.pkl".format(prefix)
        vocab_path_js = "{}/wikibio_vocab_pt.json".format(prefix)
        if self.data_src == 'train':
            print("saving vocab ...")
            self.vocab = Vocabulary(corpus=total, field=total_field, dec_type=self.dec_type)
            data = {
                "idx2word": self.vocab.idx2word,
                "word2idx": self.vocab.word2idx,
                "idx2field": self.vocab.idx2field,
                "field2idx": self.vocab.field2idx,
            }
            with open(vocab_path_pkl, 'wb') as fout:
                pickle.dump(data, fout)
            with io.open(vocab_path_js, 'w', encoding='utf-8') as fout:
                json.dump(data, fout, sort_keys=True, indent=4)
        else:
            print("loading vocab ...")
            with open(vocab_path_pkl, 'rb') as fin:
                data = pickle.load(fin)
            self.vocab = Vocabulary(word2idx=data["word2idx"], idx2word=data["idx2word"],
                                    field2idx=data["field2idx"], idx2field=data["idx2field"],
                                    dec_type=self.dec_type)
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
        if self.dec_type == 'pt':
            batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t = [], [], [], []
        source_len, target_len, w2fs = [], [], []
        list_oovs = []
        targets = []
        sources = []
        fields = []
        max_source_oov = 0
        for data in sample:
            # print("data: {}".format(data))
            source = data[0]
            target = data[1]  # (value_t, field_t, p_for_t, p_bck_t, lab_t)
            field = data[2]
            p_for = data[3]
            p_bck = data[4]
            table = data[5]
            src_len = len(source)
            if self.dec_type == 'pt':
                value_t, field_t, p_for_t, p_bck_t, lab_t = target
                p_for.append(1)
                p_bck.append(1)
                p_for_t = [1] + p_for_t + [1]
                p_bck_t = [1] + p_bck_t + [1]
                lab_t.append(src_len)  # <EOS>
                src_len += 1  # <EOS>
                tgt_len = len(value_t) + 2
            else:
                tgt_len = len(target) + 2

            # print("src_len: {}".format(src_len))
            # print("tgt_len: {}".format(tgt_len))

            source_len.append(src_len)
            target_len.append(tgt_len)

            # ----------------------- word to ids ------------------------- #
            _fields = self.vocab.vectorize_field(field)
            _o_source, source_oov, _source, oov_freq_src = self.vocab.vectorize_source(source, table)
            if self.dec_type == 'pt':
                _o_target, _target, oov_freq_tgt = self.vocab.vectorize_target(value_t, source_oov, table)
                _fields_t = self.vocab.vectorize_field(field_t, is_target=True)
            else:
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
                # w2f = {(idx+self.vocab.size): self.vocab.word2idx['<UNK>'] for word, idx in source_oov}
                w2f = {(idx + self.vocab.size): self.vocab.word2idx.get(table[word], self.vocab.word2idx['<UNK>'])
                       for word, idx in source_oov}
                w2fs.append(w2f)
                list_oovs.append(idx2word_oov)
                if self.dec_type == 'pt':
                    targets.append(value_t)  # tokens
                else:
                    targets.append(target)  # tokens
                sources.append(source)  # tokens
                fields.append(field)    # tokens

            # print("_o_source ({}): {}".format(len(_o_source), _o_source))
            # print("_fields ({}): {}".format(len(_fields), _fields))
            # print("p_for ({}): {}".format(len(p_for), p_for))
            # print("p_bck ({}): {}".format(len(p_bck), p_bck))
            # print("_source ({}): {}".format(len(_source), _source))
            #
            # print("_o_target ({}): {}".format(len(_o_target), _o_target))
            # print("_fields_t ({}): {}".format(len(_fields_t), _fields_t))
            # print("p_for_t ({}): {}".format(len(p_for_t), p_for_t))
            # print("p_bck_t ({}): {}".format(len(p_bck_t), p_bck_t))
            # print("_target ({}): {}".format(len(_target), _target))
            # print("lab_t ({}): {}".format(len(lab_t), lab_t))
            # sys.exit(0)

            batch_s.append(_source)
            batch_f.append(_fields)
            batch_pf.append(p_for)
            batch_pb.append(p_bck)
            batch_o_s.append(_o_source)

            batch_t.append(_target)
            batch_o_t.append(_o_target)
            if self.dec_type == 'pt':
                batch_f_t.append(_fields_t)
                batch_pf_t.append(p_for_t)
                batch_pb_t.append(p_bck_t)
                batch_lab_t.append(lab_t)

        batch_s = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_s]
        batch_f = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_f]
        batch_pf = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_pf]
        batch_pb = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_pb]
        batch_o_s = [torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_o_s]

        batch_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_t]
        batch_o_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_o_t]
        if self.dec_type == 'pt':
            batch_f_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_f_t]
            batch_pf_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_pf_t]
            batch_pb_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_pb_t]
            batch_lab_t = [torch.LongTensor(self.pad_vector(i, max(target_len))) for i in batch_lab_t]

        batch_s = torch.stack(batch_s, dim=0)
        batch_f = torch.stack(batch_f, dim=0)
        batch_pf = torch.stack(batch_pf, dim=0)
        batch_pb = torch.stack(batch_pb, dim=0)
        batch_o_s = torch.stack(batch_o_s, dim=0)

        batch_o_t = torch.stack(batch_o_t, dim=0)
        batch_t = torch.stack(batch_t, dim=0)
        if self.dec_type == 'pt':
            batch_f_t = torch.stack(batch_f_t, dim=0)
            batch_pf_t = torch.stack(batch_pf_t, dim=0)
            batch_pb_t = torch.stack(batch_pb_t, dim=0)
            batch_lab_t = torch.stack(batch_lab_t, dim=0)
            batch_t = (batch_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t)  # NOTE: batch_t is now a tuple

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
            if self.dec_type == 'pt':
                batch_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t = batch_t
                batch_t = batch_t.to(self.device)
                batch_f_t = batch_f_t.to(self.device)
                batch_pf_t = batch_pf_t.to(self.device)
                batch_pb_t = batch_pb_t.to(self.device)
                batch_lab_t = batch_lab_t.to(self.device)
                # batch_t = (batch_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t)  # NOTE: batch_t is now a tuple
                batch_t = (batch_t, batch_f_t, batch_lab_t)  # NOTE: batch_t is now a tuple
            else:
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

