import pickle
import os
import collections
import sys
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

sys.path.append('pycocoevalcap')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
# from pycocoevalcap.meteor.meteor import Meteor
# from pycocoevalcap.cider.cider import Cider

class Metrics(object):
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            # (Meteor(), "METEOR"),
            # (Cider(), "CIDEr")
            ]
        self.fields = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L"]

    def convert(self, data):
        if isinstance(data, basestring):
            return data.encode('utf-8')
        elif isinstance(data, collections.Mapping):
            return dict(map(convert, data.items()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(convert, data))
        else:
            return data

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            print("Metric: {}".format(method))
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores

    def compute_metrics(self, get_scores=True, live=False, **kwargs):
        pred = kwargs.pop('cands_ids', {})
        gold = kwargs.pop('tgts_ids', {})
        if live:
            temp_ref = kwargs.pop('ref', {})
            cand = kwargs.pop('cand', {})
        else:
            reference_path = kwargs.pop('ref', '')
            candidate_path = kwargs.pop('cand', '')
            with open(reference_path, 'rb') as f:
                temp_ref = pickle.load(f)
            with open(candidate_path, 'rb') as f:
                cand = pickle.load(f)

        epoch = kwargs.pop('epoch', 0)
        # make dictionary
        hypo = {}
        ref = {}
        i = 0
        for vid, caption in cand.items():
            hypo[i] = [caption]
            ref[i] = temp_ref[vid]
            i += 1

        print("Computing Scores ...")
        final_scores = self.score(ref, hypo)
        final_scores = self.non_rg_metrics(pred, gold, final_scores)
        for k, v in final_scores.items():
            print('[epoch-{}]{}:\t{}'.format(epoch, k, v))

        if get_scores:
            return final_scores

    def remove_dups(self, seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def non_rg_metrics(self, pred, gold, final_scores):
        print("Computing F1 ...")
        try:
            assert len(pred) == len(gold)
        except AssertionError:
            raise ValueError("len(pred) = {}; len(gold) = {}".format(len(pred), len(gold)))

        print("{} pairs to be evaluated".format(len(pred)))

        true_positives, predicted, total_gold = 0, 0, 0
        ndld = 0.0

        for i in range(1, len(pred)+1):
            ascii_start = 1
            p = self.remove_dups(pred[i])
            g = self.remove_dups(gold[i])
            s1 = ''.join([chr(ascii_start + i) for i in range(len(p))])
            pred_dict = {n: s for n,s in zip(p, s1)}
            s2 = ''
            next_char = ascii_start + len(s1)

            for x in g:
                if x in pred_dict:
                    s2 += pred_dict[x]
                else:
                    s2 += chr(next_char)
                    next_char += 1

            gold_dict = dict.fromkeys(g)
            tp = [1 if x in gold_dict else 0 for x in p]
            tp = sum(tp)
            true_positives += tp
            predicted += len(p)
            total_gold += len(g)
            ndld += 100.0*(1 - normalized_damerau_levenshtein_distance(s1, s2))

        precision = 100.0*float(true_positives) / predicted
        recall = 100.0*float(true_positives) / total_gold
        f1 = 2*precision*recall/(precision+recall)
        ndld /= len(pred)

        final_scores['precision'] = precision
        final_scores['recall'] = recall
        final_scores['f1'] = f1
        final_scores['ndld'] = ndld

        return final_scores


if __name__ == '__main__':
    cand = {'generated_description1': 'how are you', 'generated_description2': 'Hello how are you'}
    ref = {'generated_description1': ['what are you', 'where are you'],
           'generated_description2': ['Hello how are you', 'Hello how is your day']}
    print(normalized_damerau_levenshtein_distance('1234', '3150'))
    cands_ids = {1: [1, 2, 3, 4]}
    tgts_ids = {1: [3, 1, 5, 0]}
    x = Metrics()
    x.compute_metrics(get_scores=False, live=True, cand=cand, ref=ref, cands_ids=cands_ids, tgts_ids=tgts_ids)
