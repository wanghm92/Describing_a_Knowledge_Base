import torch
from collections import Counter
import pickle, sys, json, io, copy
from os.path import expanduser
HOME = expanduser("~")
from tqdm import tqdm
import numpy as np
from utils.content_metrics import Content_Metrics

class Vocabulary:
    """Vocabulary class for mapping between words and ids"""
    def __init__(self,
                 word2idx=None, idx2word=None,
                 field2idx=None, idx2field=None,
                 rcd2idx=None, idx2rcd=None,
                 ha2idx=None, idx2ha=None,
                 field=None, corpus=None,
                 start_end_tokens=True,
                 max_words=20000,
                 min_frequency=0, min_frequency_field=0, min_frequency_rcd=0,
                 pad_id=0, sos_id=1, eos_id=2, unk_id=3,
                 dec_type='pt'):

        self.dec_type = dec_type
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

        if corpus is None:
            self.word2idx = word2idx
            self.idx2word = idx2word
            self.field2idx = field2idx
            self.idx2field = idx2field
            self.rcd2idx = rcd2idx
            self.idx2rcd = idx2rcd
            self.ha2idx = ha2idx
            self.idx2ha = idx2ha

            self.start_end_tokens = False
            self.size = len(word2idx)
            self.field_vocab_size = len(field2idx)
            self.rcd_vocab_size = len(self.rcd2idx)
            self.ha_vocab_size = len(self.ha2idx)

        else:
            self.word2idx = dict()
            self.idx2word = dict()
            self.size = 0
            self.vocabulary = None

            self.field2idx = dict()
            self.idx2field = dict()
            self.field_vocab_size = 0
            self.field_vocab = None

            self.rcd2idx = dict()
            self.idx2rcd = dict()
            self.rcd_vocab_size = 0
            self.rcd_vocab = None

            self.ha2idx = dict()
            self.idx2ha = dict()
            self.ha_vocab_size = 0

            # TODO: check the min_frequency
            # most common words
            self.max_words = max_words
            # least common words
            self.min_frequency = min_frequency
            self.min_frequency_field = min_frequency_field
            self.min_frequency_rcd = min_frequency_rcd

            self.start_end_tokens = start_end_tokens

            self._build_vocabulary(corpus, field)
            print("Finish build vocabulary")

            self._build_word_index()
            print("Finish build word index")

            self.size = len(self.word2idx)
            self.field_vocab_size = len(self.field2idx)
            self.rcd_vocab_size = len(self.rcd2idx)
            self.ha_vocab_size = len(self.ha2idx)

            # print(self.field_vocab)
            print("Record Type vocab:")
            print(self.rcd_vocab)
            print("Vocab sizes:")
            print(self.size)
            print(self.field_vocab_size)
            print(self.rcd_vocab_size)
            print(self.ha_vocab_size)

    def _build_vocabulary(self, corpus, field):
        assert isinstance(field, tuple)
        field, rcd = field

        vocabulary = Counter(word for sent in corpus for word in sent)
        field_vocab = Counter(word for sent in field for word in sent)
        rcd_vocab = Counter(word for sent in rcd for word in sent)

        if self.max_words:
            vocabulary = {word: freq for word,
                          freq in vocabulary.most_common(self.max_words)}
        if self.min_frequency:
            vocabulary = {word: freq for word, freq in vocabulary.items()
                          if freq >= self.min_frequency}

        if self.min_frequency_field:
            field_vocab = {word: freq for word, freq in field_vocab.items()
                           if freq >= self.min_frequency_field}

        if self.min_frequency_rcd:
            rcd_vocab = {word: freq for word, freq in rcd_vocab.items()
                              if freq >= self.min_frequency_rcd}

        self.vocabulary = Counter(vocabulary)
        self.field_vocab = Counter(field_vocab)
        self.rcd_vocab = Counter(rcd_vocab)

    def _build_luts(self, vocab):
        tk2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        offset = len(tk2idx)
        for idx, word in enumerate(vocab):
            tk2idx[word] = idx + offset
        idx2tk = {idx: word for word, idx in tk2idx.items()}
        return tk2idx, idx2tk

    def _build_word_index(self):
        """ build tk2idx and idx2tk look-up tables"""
        # word idx
        self.word2idx, self.idx2word = self._build_luts(self.vocabulary)
        # field idx
        self.field2idx, self.idx2field = self._build_luts(self.field_vocab)
        # record type idx
        self.rcd2idx, self.idx2rcd = self._build_luts(self.rcd_vocab)
        # home/away idx
        self.ha2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'HOME': 4, 'AWAY': 5}
        self.idx2ha = {idx: h for h, idx in self.ha2idx.items()}

    def add_start_end(self, vector, item2idx=None):
        """ prepend <SOS> and append <EOS> for sequences"""
        vector.append(item2idx['<EOS>'])
        return [item2idx['<SOS>']] + vector

    def vectorize_feature(self, vector, ft_type='field', is_outline=False):

        if ft_type == 'field':
            ft2idx = self.field2idx
        elif ft_type == 'rcd':
            ft2idx = self.rcd2idx
        elif ft_type == 'ha':
            ft2idx = self.ha2idx
        else:
            raise ValueError("{} not supported".format(ft_type))

        _ft = [ft2idx.get(x, ft2idx['<UNK>']) for x in vector]
        if self.dec_type in ['pt', 'prn'] or is_outline:
            # add <SOS> at the beginning and <EOS> to the end for ptr-net to choose from
            _ft = self.add_start_end(_ft, item2idx=ft2idx)
        return _ft

    def vectorize_source(self, vector):
        """ source word to idx"""
        _oov = {}
        _o_source, _source = [], []
        cnt = 0
        oov_freq = 0
        for word in vector:
            try:
                _o_source.append(self.word2idx[word])
                _source.append(self.word2idx[word])
            except KeyError:
                oov_freq += 1
                if self.dec_type in ['pg', 'prn']:
                    if word not in _oov:
                        _o_source.append(cnt + self.size)
                        _oov[word] = cnt
                        cnt += 1
                    else:
                        _o_source.append(_oov[word] + self.size)
                else:
                    _o_source.append(self.word2idx['<UNK>'])
                _source.append(self.word2idx['<UNK>'])
                # use field type embedding for OOV values
                # _source.append(self.word2idx.get(table[word], self.word2idx['<UNK>']))

        # add <SOS> at the beginning and <EOS> to the end for ptr-net to choose from
        if self.dec_type in ['pt', 'prn']:
            # _o_source.append(self.word2idx['<EOS>'])
            # _source.append(self.word2idx['<EOS>'])
            _o_source = self.add_start_end(_o_source, item2idx=self.word2idx)
            _source = self.add_start_end(_source, item2idx=self.word2idx)
        return _o_source, _oov, _source, oov_freq

    def vectorize_outline(self, vector):
        """ outline tokens to idx"""
        _oov = {}
        cnt = 0
        _o_outline, _outline = [], []
        oov_freq = 0
        for word in vector:
            try:
                _o_outline.append(self.word2idx[word])
                _outline.append(self.word2idx[word])
            except KeyError:
                oov_freq += 1
                if self.dec_type in ['pt', 'prn']:
                    if word not in _oov:
                        _o_outline.append(cnt + self.size)
                        _oov[word] = cnt
                        cnt += 1
                    else:
                        _o_outline.append(_oov[word] + self.size)
                else:
                    _o_outline.append(self.word2idx['<UNK>'])
                _outline.append(self.word2idx['<UNK>'])

        return self.add_start_end(_o_outline, self.word2idx), _oov, self.add_start_end(_outline, self.word2idx), oov_freq

    def vectorize_summary(self, vector, tail_oov_vocab=None):
        """ summary word to idx"""
        _o_summary, _summary = [], []
        oov_freq = 0
        for word in vector:
            try:
                _o_summary.append(self.word2idx[word])
                _summary.append(self.word2idx[word])
            except KeyError:
                oov_freq += 1
                if tail_oov_vocab:
                    if word not in tail_oov_vocab:
                        _o_summary.append(self.word2idx['<UNK>'])
                        _summary.append(self.word2idx['<UNK>'])
                    else:
                        _o_summary.append(tail_oov_vocab[word] + self.size)
                        _summary.append(self.word2idx['<UNK>'])
                else:
                    _o_summary.append(self.word2idx['<UNK>'])
                    _summary.append(self.word2idx['<UNK>'])

        return self.add_start_end(_o_summary, self.word2idx), self.add_start_end(_summary, self.word2idx), oov_freq


class Table2text_seq:
    def __init__(self, data_src, type=0, batch_size=128, USE_CUDA=torch.cuda.is_available(),
                 train_mode=False, dec_type='pg'):
        # TODO: change the path
        prefix = "{}/table2text_nlg/data/dkb/rotowire_prn/".format(HOME)

        assert type == 3
        self.vocab = None
        self.dec_type = dec_type
        self.text_len = 0
        self.data_src = data_src
        self.batch_size = batch_size
        self.USE_CUDA = USE_CUDA
        self.device = torch.device("cuda" if USE_CUDA else "cpu")

        self.oov_cnt_src = 0
        self.oov_cnt_otl = 0
        self.oov_cnt_sum = 0
        self.total_cnt_src = 0
        self.total_cnt_otl = 0
        self.total_cnt_sum = 0

        # ----------------------- file names ------------------------- #
        if data_src == 'train' or data_src == 'train4eval':
            path = "{}train_P.pkl".format(prefix)
        elif data_src == 'valid':
            path = "{}valid_P.pkl".format(prefix)
        elif data_src == 'test':
            path = "{}test_P.pkl".format(prefix)
        else:
            raise ValueError("Only train, valid, test data_srcs are supported")

        # ----------------------- load triples and build vocabulary ------------------------- #
        self.prepare_content_metrics()
        if data_src == 'train' and not train_mode:
            self.load_vocab(path)
        else:
            self.data = self.load_data(path)
            self.len = len(self.data)
            # ----------------------- word to ids ------------------------- #
            self.corpus = self.batchfy()

        print("vocab size = {}".format(self.vocab.size))

    def prepare_content_metrics(self):
        domain = 'train' if 'train' in self.data_src else self.data_src
        self.content_metrics = Content_Metrics()
        self.goldfi = "{}/table2text_nlg/harvardnlp/data2text-plan-py/rotowire/{}/roto-gold-{}.h5-tuples.txt".format(HOME, domain, domain)
        self.gold_triples = self._get_gold_triples()[1:]

        self.src_file = "{}/table2text_nlg/harvardnlp/data2text-plan-py/rotowire/{}/src_{}.txt".format(HOME, domain, domain)
        inputs = []
        with io.open(self.src_file, 'r', encoding="utf-8") as fin:
            for _, line in enumerate(fin):
                inputs.append(line.split())
        self.inputs = inputs

    def _get_gold_triples(self):
        all_triples = []
        curr = []
        with open(self.goldfi) as f:
            for line in f:
                if line.isspace():
                    all_triples.append(self.content_metrics.dedup(curr))
                    curr = []
                else:
                    pieces = line.strip().split('|')
                    pieces = [pieces[0].lower()] + pieces[1:]
                    curr.append(tuple(pieces))
        if len(curr) > 0:
            all_triples.append(self.content_metrics.dedup(curr))
        return all_triples

    def _get_pred_triples(self, pred_ids):

        DELIM = u"￨"

        eval_outputs = []
        for i, sample in tqdm(enumerate(self.inputs)):
            content_plan = pred_ids[i]
            eval_output = []
            for record in content_plan:
                elements = sample[int(record-1)].split(DELIM)
                if elements[0].isdigit():
                    record_type = elements[2]
                    if not elements[2].startswith('TEAM'):
                        record_type = 'PLAYER-' + record_type
                    eval_output.append((elements[1].replace("_", " ").strip('<').strip('>').lower(), elements[0], record_type))
            eval_outputs.append(eval_output)

        return eval_outputs

    def load_vocab(self, path):
        print("Loading data ** LIGHT ** from {}".format(path))
        prefix = "{}/table2text_nlg/describe_kb/outputs".format(HOME)
        vocab_path_pkl = "{}/rotowire_vocab_prn.pkl".format(prefix)
        print("loading vocab ... from {}".format(vocab_path_pkl))
        with open(vocab_path_pkl, 'rb') as fin:
            data = pickle.load(fin)

        # TODO: include record type and H/A vocabs
        self.vocab = Vocabulary(word2idx=data["word2idx"], idx2word=data["idx2word"],
                                field2idx=data["field2idx"], idx2field=data["idx2field"],
                                rcd2idx=data["rcd2idx"], idx2rcd=data["idx2rcd"],
                                ha2idx=data["ha2idx"], idx2ha=data["idx2ha"],
                                dec_type=self.dec_type)

    def load_data(self, path):

        # TODO: add TEAM/PLAYER feature

        prefix = "{}/table2text_nlg/describe_kb/outputs".format(HOME)
        print("Loading data from {}".format(path))
        # (qkey, qitem, index)
        with open(path, 'rb') as fin:
            data = pickle.load(fin)
        old_sources = data["source"]
        old_outlines = data["outlines"]
        old_summaries = data["summaries"]
        samples = []
        total = []
        total_field = []
        total_rcd = []

        print("{} samples to be processed".format(len(old_sources)))
        for idx, old_source in enumerate(tqdm(old_sources)):
            table = {}
            value_s = []
            field_s = []
            rcd_s = []
            ha_s = []

            for key, value, rcd, ha in old_source:
                value = value.lower()  # NOTE: changed to lowercase strings
                key = key.lower()  # NOTE: changed to lowercase strings
                tag = '<'+key+'>'  # NOTE: change key into special tokens
                value_s.append(value)
                field_s.append(tag)
                rcd_s.append(rcd)
                ha_s.append(ha)

                if value not in table:
                    table[value] = tag

            old_outline = old_outlines[idx]
            temp = copy.deepcopy(old_outline)
            if len(list(temp)) > self.text_len:
                self.text_len = len(list(temp)) + 2

            value_t = []
            field_t = []
            rcd_t = []
            ha_t = []
            lab_t = []
            for key, value, rcd, ha, lab in old_outline:
                value = value.lower()  # NOTE: changed to lowercase strings
                key = key.lower()  # NOTE: changed to lowercase strings
                tag = '<'+key+'>'  # NOTE: change key into special tokens
                value_t.append(value)
                field_t.append(tag)
                rcd_t.append(rcd)
                ha_t.append(ha)
                lab_t.append(lab+1)  # +1 for sources have leading <sos>; type is int

            summary = old_summaries[idx]
            summary = [x.lower() for x in summary]  # NOTE: changed to lowercase strings

            # print("value_s: \n{}".format(value_s))
            # print("field_s: \n{}".format(field_s))
            # print("rcd_s: \n{}".format(rcd_s))
            # print("ha_s: \n{}".format(ha_s))
            # print("value_t: \n{}".format(value_t))
            # print("field_t: \n{}".format(field_t))
            # print("rcd_t: \n{}".format(rcd_t))
            # print("ha_t: \n{}".format(ha_t))
            # print("lab_t: \n{}".format(lab_t))
            # print("summary: \n{}".format(summary))
            outline = (value_t, field_t, rcd_t, ha_t, lab_t)
            total.append(value_s + value_t + summary)
            total_field.append(field_s + field_t)
            total_rcd.append(rcd_s + rcd_t)
            samples.append([value_s, outline, field_s, rcd_s, ha_s, table, summary])

            # print("total: \n{}".format(total))

        '''
            torch.nn.utils.rnn.pack_padded_sequence requires the sequence lengths sorted in decreasing order
        '''
        print("sorting samples ...")
        # self.sort_indices = list(range(len(samples)))
        # self.sort_indices = np.argsort([len(x[0]) for x in samples]).tolist()
        self.sort_indices = np.argsort([len(x[1][0]) for x in samples]).tolist()  # increasing length
        self.sort_indices.reverse()  # decreasing length
        samples = np.array(samples)[self.sort_indices].tolist()

        vocab_path_pkl = "{}/rotowire_vocab_prn.pkl".format(prefix)
        vocab_path_js = "{}/rotowire_vocab_prn.json".format(prefix)
        if self.data_src == 'train':
            print("saving vocab ...")
            self.vocab = Vocabulary(corpus=total, field=(total_field, total_rcd), dec_type=self.dec_type)
            data = {
                "idx2word": self.vocab.idx2word,
                "word2idx": self.vocab.word2idx,
                "idx2field": self.vocab.idx2field,
                "field2idx": self.vocab.field2idx,
                "idx2rcd": self.vocab.idx2rcd,
                "rcd2idx": self.vocab.rcd2idx,
                "idx2ha": self.vocab.idx2ha,
                "ha2idx": self.vocab.ha2idx,
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
                                    rcd2idx=data["rcd2idx"], idx2rcd=data["idx2rcd"],
                                    ha2idx=data["ha2idx"], idx2ha=data["idx2ha"],
                                    dec_type=self.dec_type)
        return samples

    def batchfy(self):
        print("Constructing Batches ...")
        samples = [self.data[i:i+self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        corpus = []
        for sample in tqdm(samples):
            corpus.append(self.vectorize(sample))
        print("oov_cnt_src: {0} ({1:.3f}%)".format(self.oov_cnt_src, 100.0*self.oov_cnt_src/self.total_cnt_src))
        print("oov_cnt_otl: {0} ({1:.3f}%)".format(self.oov_cnt_otl, 100.0*self.oov_cnt_otl/self.total_cnt_otl))
        print("oov_cnt_sum: {0} ({1:.3f}%)".format(self.oov_cnt_sum, 100.0*self.oov_cnt_sum/self.total_cnt_sum))
        print("total_cnt_src: {}".format(self.total_cnt_src))
        print("total_cnt_otl: {}".format(self.total_cnt_otl))
        print("total_cnt_sum: {}".format(self.total_cnt_sum))
        return corpus

    def pad_vector(self, vector, maxlen):
        padding = maxlen - len(vector)
        vector.extend([0] * padding)
        return vector

    def pad_vec_rev(self, vector, maxlen):
        vector = vector[::-1][1:-1]
        padding = maxlen - len(vector)
        vector.extend([0] * padding)
        return vector

    def vectorize(self, sample):
        """
            batch_s         --> tensor batch of table with ids
            batch_o_s       --> tensor batch of table with ids and <unk> replaced by temp OOV ids
            batch_t         --> tensor batch of text with ids
            batch_o_t       --> tensor batch of outline and <unk> replaced by temp OOV ids
            batch_f         --> tensor batch of field with ids(might not exist)
            batch_pf        --> tensor batch of forward position
            batch_pb        --> tensor batch of backward position
            batch_o_f       --> tensor batch of field and used wordid
            max_article_oov --> max number of OOV tokens in article batch
        """

        batch_o_s, batch_s, batch_f, batch_pf, batch_pb = [], [], [], [], []
        batch_o_t, batch_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t = [], [], [], [], [], []
        batch_sum, batch_o_sum = [], []
        source_len, outline_len, summary_len, w2fs = [], [], [], []
        sources, fields, rcds, has, outlines, summaries = [], [], [], [], [], []
        batch_idx2oov = []
        max_tail_oov = 0
        for data in sample:
            # print("data: {}".format(data))
            # data: [value_s, outline, field_s, rcd_s, ha_s, table, summary]
            source = data[0]
            outline = data[1]  # (value_t, field_t, rcd_t, ha_t, lab_t)
            field = data[2]
            rcd = data[3]
            ha = data[4]
            table = data[5]
            summary = data[6]
            value_t, field_t, rcd_t, ha_t, lab_t = outline  #tokens

            # <EOS> and <SOS>
            src_len = len(source)
            sum_len = len(summary) + 2
            otl_len = len(value_t) + 2
            if self.dec_type in ['pt', 'prn']:
                src_len += 2

            # print("src_len: {}".format(src_len))
            # print("otl_len: {}".format(otl_len))
            # print("sum_len: {}".format(sum_len))

            source_len.append(src_len)
            outline_len.append(otl_len)
            summary_len.append(sum_len)

            # ----------------------- source word and features to ids ------------------------- #
            _o_source, source_oov, _source, oov_freq_src = self.vocab.vectorize_source(source)
            _fields = self.vocab.vectorize_feature(field, ft_type='field')
            _rcd = self.vocab.vectorize_feature(rcd, ft_type='rcd')
            _ha = self.vocab.vectorize_feature(ha, ft_type='ha')

            # ----------------------- outline word and features to ids ------------------------- #
            _o_outline, outline_oov, _outline, oov_freq_otl = self.vocab.vectorize_outline(value_t)
            _fields_t = self.vocab.vectorize_feature(field_t, ft_type='field', is_outline=True)
            _rcd_t = self.vocab.vectorize_feature(rcd_t, ft_type='rcd', is_outline=True)
            _ha_t = self.vocab.vectorize_feature(ha_t, ft_type='ha', is_outline=True)
            _lab_t = [0] + lab_t + [src_len-1]  # start from copying the 0th <SOS> token

            # ----------------------- summary word to ids ------------------------- #
            if self.dec_type in ['pg', 'prn']:
                tail_oov_vocab = source_oov
            else:
                tail_oov_vocab = None

            _o_summary, _summary, oov_freq_sum = self.vocab.vectorize_summary(summary, tail_oov_vocab)

            # ----------------------- update oov stats ------------------------- #
            self.oov_cnt_src += oov_freq_src
            self.oov_cnt_otl += oov_freq_otl
            self.oov_cnt_sum += oov_freq_sum
            self.total_cnt_src += len(_source)
            self.total_cnt_otl += len(_outline)
            self.total_cnt_sum += len(_summary)

            # ----------------------- prepare for evaluation ------------------------- #
            if tail_oov_vocab:
                tail_oov_vocab = tail_oov_vocab.items()
                if max_tail_oov < len(tail_oov_vocab):
                    max_tail_oov = len(tail_oov_vocab)
                idx2oov = {idx: word for word, idx in tail_oov_vocab}
                w2f = {(idx+self.vocab.size): self.vocab.word2idx['<UNK>'] for word, idx in tail_oov_vocab}
                # w2f = {(idx + self.vocab.size): self.vocab.word2idx.get(table[word], self.vocab.word2idx['<UNK>'])
                #        for word, idx in tail_oov_vocab}
            else:
                idx2oov = {}
                w2f = {}

            w2fs.append(w2f)
            batch_idx2oov.append(idx2oov)

            if self.data_src != 'train':
                sources.append(source)
                fields.append(field)
                rcds.append(rcd)
                has.append(ha)
                outlines.append(value_t)
                summaries.append(summary)

            # print("_source ({}): {}".format(len(_source), _source))
            # print("_fields ({}): {}".format(len(_fields), _fields))
            # print("_rcd ({}): {}".format(len(_rcd), _rcd))
            # print("_ha ({}): {}".format(len(_ha), _ha))
            # print("_o_source ({}): {}".format(len(_o_source), _o_source))
            # print(src_len)
            assert len(_source) == len(_fields) == len(_rcd) == len(_ha) == len(_o_source) == src_len
            batch_s.append(_source)
            batch_f.append(_fields)
            batch_pf.append(_rcd)
            batch_pb.append(_ha)
            batch_o_s.append(_o_source)  # for scatter add

            # print("_outline ({}): {}".format(len(_outline), _outline))
            # print("_fields_t ({}): {}".format(len(_fields_t), _fields_t))
            # print("_rcd_t ({}): {}".format(len(_rcd_t), _rcd_t))
            # print("_ha_t ({}): {}".format(len(_ha_t), _ha_t))
            # print("_lab_t ({}): {}".format(len(_lab_t), _lab_t))
            # print("_o_outline ({}): {}".format(len(_o_outline), _o_outline))
            # print(otl_len)
            assert len(_outline) == len(_fields_t) == len(_rcd_t) == len(_ha_t) == len(_lab_t) == len(_o_outline) == otl_len
            batch_t.append(_outline)
            batch_f_t.append(_fields_t)
            batch_pf_t.append(_rcd_t)
            batch_pb_t.append(_ha_t)
            batch_o_t.append(_outline)  # for scatter add attn weights
            batch_lab_t.append(_lab_t)  # for CrossEntropyLoss

            # print("_summary ({}): {}".format(len(_summary), _summary))
            # print("_o_summary ({}): {}".format(len(_o_summary), _o_summary))
            assert len(_summary) == len(_o_summary) == sum_len
            batch_sum.append(_summary)
            batch_o_sum.append(_o_summary)  # for gather NLL loss

        # ----------------------- convert to list of tensors and pad to max length ------------------------- #
        batch_s = torch.stack([torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_s], dim=0)
        batch_o_s = torch.stack([torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_o_s], dim=0)
        batch_f = torch.stack([torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_f], dim=0)
        batch_pf = torch.stack([torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_pf], dim=0)
        batch_pb = torch.stack([torch.LongTensor(self.pad_vector(i, max(source_len))) for i in batch_pb], dim=0)

        source_package = (batch_s, batch_o_s, batch_f, batch_pf, batch_pb)

        if self.dec_type in ['pt', 'prn']:
            batch_t = torch.stack([torch.LongTensor(self.pad_vector(i, max(outline_len))) for i in batch_t], dim=0)
            batch_o_t = torch.stack([torch.LongTensor(self.pad_vector(i, max(outline_len))) for i in batch_o_t], dim=0)
            batch_f_t = torch.stack([torch.LongTensor(self.pad_vector(i, max(outline_len))) for i in batch_f_t], dim=0)
            batch_pf_t = torch.stack([torch.LongTensor(self.pad_vector(i, max(outline_len))) for i in batch_pf_t], dim=0)
            batch_pb_t = torch.stack([torch.LongTensor(self.pad_vector(i, max(outline_len))) for i in batch_pb_t], dim=0)
            batch_lab_t = torch.stack([torch.LongTensor(self.pad_vector(i, max(outline_len))) for i in batch_lab_t], dim=0)

            outline_package = [batch_t, batch_o_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t]
            # if self.dec_type == 'prn':
            #     batch_t_r = torch.stack([torch.LongTensor(self.pad_vec_rev(i, max(outline_len))) for i in batch_t], dim=0)  # for emb lookup
            #     batch_o_t_r = torch.stack([torch.LongTensor(self.pad_vec_rev(i, max(outline_len))) for i in batch_o_t], dim=0)  # for scatter add attn weights
            #     batch_f_t_r = torch.stack([torch.LongTensor(self.pad_vec_rev(i, max(outline_len))) for i in batch_f_t], dim=0)
            #     batch_pf_t_r = torch.stack([torch.LongTensor(self.pad_vec_rev(i, max(outline_len))) for i in batch_pf_t], dim=0)
            #     batch_pb_t_r = torch.stack([torch.LongTensor(self.pad_vec_rev(i, max(outline_len))) for i in batch_pb_t], dim=0)
            #     outline_pkg_rev = [batch_t_r, batch_o_t_r, batch_f_t_r, batch_pf_t_r, batch_pb_t_r]
            # else:
            outline_pkg_rev = None
        else:
            outline_pkg_rev = None
            outline_package = None

        if self.dec_type != 'pt':
            batch_sum = torch.stack([torch.LongTensor(self.pad_vector(i, max(summary_len))) for i in batch_sum], dim=0)
            batch_o_sum = torch.stack([torch.LongTensor(self.pad_vector(i, max(summary_len))) for i in batch_o_sum], dim=0)

            summary_package = (batch_sum, batch_o_sum)
        else:
            summary_package = None

        sources = [i[:max(source_len)] for i in sources]
        fields = [i[:max(source_len)] for i in fields]
        rcds = [i[:max(source_len)] for i in rcds]
        has = [i[:max(source_len)] for i in has]
        fields = {'fields': fields, 'rcds': rcds, 'has': has}
        outlines = [i[:max(outline_len)-2] for i in outlines]
        summaries = [i[:max(summary_len)-2] for i in summaries]
        outline_len = [x-1 for x in outline_len]  # minus the <EOS> token for passing to encoder_otl

        texts_package = [sources, fields, summaries, outlines]

        remaining = [source_len, outline_len, summary_len, max_tail_oov, w2fs, batch_idx2oov]

        return source_package, outline_package, outline_pkg_rev, summary_package, texts_package, remaining

    def dump_to_device(self, data_packages):
        return [[t.to(self.device) for t in pkg] if pkg is not None else None for pkg in data_packages]

    def get_batch(self, index):

        source_package, outline_package, outline_pkg_rev, summary_package, texts_package, remaining = self.corpus[index]

        data_packages = [source_package, outline_package, outline_pkg_rev, summary_package]

        data_packages = self.dump_to_device(data_packages)

        return data_packages, texts_package, remaining

        # batch_s, batch_o_s, batch_f, batch_f, batch_pf, batch_pb = source_package
        # batch_t, batch_o_t, batch_f_t, batch_pf_t, batch_pb_t, batch_lab_t = outline_package
        # batch_sum, batch_o_sum = summary_package
        # source_len, max_tail_oov, w2fs, list_oovs, sources, fields, summaries, outlines = others
        #
        #
        # if self.data_src == 'train':
        #     return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, batch_t, batch_o_t, source_len, max_tail_oov
        # else:
        #     return batch_s, batch_o_s, batch_f, batch_pf, batch_pb, batch_t, batch_o_t, source_len, max_tail_oov, \
        #            w2fs, sources, targets, fields, list_oovs

