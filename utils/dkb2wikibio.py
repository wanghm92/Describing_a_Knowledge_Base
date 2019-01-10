import torch
from collections import Counter
import pickle, sys, json, io, argparse
from os.path import expanduser
HOME = expanduser("~")
import spacy
from spacy.lang.en import English
nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])
tokenizer = English().Defaults.create_tokenizer(nlp)
output_path = "{}/table2text_nlg/data/dkb/wikibio_format_tokenized".format(HOME)
from tqdm import tqdm

class Vocabulary:
    """Vocabulary class for mapping between words and ids"""
    def __init__(self, word2idx=None, idx2word=None, field=None, corpus=None, max_words=50000, min_frequency=30,
                 start_end_tokens=True, min_frequency_field=100):
        if corpus is None:
            self.word2idx = word2idx
            self.idx2word = idx2word
            self.start_end_tokens = False
            self.size = len(word2idx)
        else:
            self.word2idx = dict()
            self.idx2word = dict()
            self.size = 0
            self.vocabulary = None
            # most common words
            self.max_words = max_words
            # least common words
            self.min_frequency = min_frequency
            self.min_frequency_field = min_frequency_field
            self.start_end_tokens = start_end_tokens
            self._build_vocabulary(corpus, field)
            print("Finish build vocabulary")

    def _build_vocabulary(self, corpus, field):
        vocabulary = Counter(word for sent in corpus for word in sent)
        field_vocab = Counter(word for sent in field for word in sent)
        print(len(field_vocab))
        if self.max_words:
            vocabulary = {word: freq for word,
                          freq in vocabulary.most_common(self.max_words)}
        if self.min_frequency:
            vocabulary = {word: freq for word, freq in vocabulary.items()
                          if freq >= self.min_frequency}
        if self.min_frequency_field:
            field_vocab = {word: freq for word, freq in field_vocab.items()
                           if freq >= self.min_frequency_field}
        print(len(field_vocab))
        self.vocabulary = vocabulary.keys()
        self.field_vocab = field_vocab.keys()

class Table2text_seq:
    def __init__(self, mode, type=0, batch_size=128, USE_CUDA=torch.cuda.is_available()):
        input_path = "{}/table2text_nlg/data/dkb/".format(HOME)
        self.type = type
        self.vocab = None
        # self.target_vocab = None
        self.text_len = 0
        self.max_p = 0
        self.mode = mode
        self.batch_size = batch_size
        self.USE_CUDA = USE_CUDA
        if mode == 0:
            if self.type == 0:
                path = "{}train_P.pkl".format(input_path)
            else:
                path = "{}train_A.pkl".format(input_path)
        elif mode == 1:
            if self.type == 0:
                path = "{}valid_P.pkl".format(input_path)
            else:
                path = "{}valid_A.pkl".format(input_path)
        else:
            if self.type == 0:
                path = "{}test_P.pkl".format(input_path)
            else:
                path = "{}test_A.pkl".format(input_path)

        self.data = self.load_data(path)
        if self.mode == 0:
            print(len(self.vocab.vocabulary))
            print(len(self.vocab.field_vocab))
        self.len = len(self.data)
        print(self.len)

    def load_data(self, path):
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
            source = []
            field = []
            p_for = []
            p_bck = []
            target = old_targets[idx]
            target = [str(x) for x in tokenizer(' '.join(target).lower())]  # NOTE: changed to lowercase strings
            if len(target) > self.text_len:
                self.text_len = len(target) + 2
            for key, value, index in old_source:
                value = [str(x) for x in tokenizer(value.lower())]  # NOTE: changed to lowercase strings
                source.extend(value)
                tag = '<{}>'.format('_'.join(key.split()))  # change key into special tokens
                field.extend([tag for _ in range(len(value))])
                p_for.extend([index for _ in range(len(value))])
            curr_p_max = max(p_for) + 1
            for p in p_for:
                p_bck.append(curr_p_max - p)
            if self.max_p < curr_p_max:
                self.max_p = curr_p_max
            # print("source: {}".format(source))
            # print("field: {}".format(field))
            assert len(source) == len(field) == len(p_for) == len(p_bck)
            total.append(source + target)
            total_field.append(field)
            samples.append([source, target, field, p_for, p_bck])

        print("sorting samples ...")
        samples.sort(key=lambda x: len(x[0]), reverse=True)

        if self.mode == 0:
            print("saving vocab ...")
            word_vocab_path = "{}/word_vocab.txt".format(output_path)
            field_vocab_path = "{}/field_vocab.txt".format(output_path)
            self.vocab = Vocabulary(corpus=total, field=total_field)
            with open(word_vocab_path, 'w+') as fout:
                for x in self.vocab.vocabulary:
                    fout.write("{}\n".format(x))
            with open(field_vocab_path, 'w+') as fout:
                for x in self.vocab.field_vocab:
                    fout.write("{}\n".format(x))

        return samples


def dump_dataset(dataset, prefix):
    data = dataset.data
    print("dumping {} samples".format(len(data)))
    box_path = "{}.box".format(prefix)
    summary_path = "{}.summary".format(prefix)

    with open(box_path, 'w+', encoding='utf-8') as fout_box, open(summary_path, 'w+', encoding='utf-8') as fout_summary:
        for d in data:
            source, target, field, p_for, p_bck = d
            fout_summary.write("{}\n".format(" ".join(target)))
            for f, s, p in zip(field, source, p_for):
                fout_box.write("{}_{}:{}\t".format(f, p, s))
            fout_box.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='loader')
    parser.add_argument('--type', type=int, default=0,
                        help='person(0)/animal(1)')
    args = parser.parse_args()
    
    print("Converting training data ...")
    train_dataset = Table2text_seq(0, type=args.type)
    print("number of training examples: %d" % train_dataset.len)
    dump_dataset(train_dataset, output_path+'/train')

    print("Converting valid data ...")
    v_dataset = Table2text_seq(1, type=args.type)
    print("number of valid examples: %d" % v_dataset.len)
    dump_dataset(v_dataset, output_path+'/valid')


    print("Converting test data ...")
    test_dataset = Table2text_seq(2, type=args.type)
    print("number of test examples: %d" % test_dataset.len)
    dump_dataset(v_dataset, output_path+'/test')

