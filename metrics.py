import pickle
import os
import collections
import sys
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from utils.content_metrics import Content_Metrics
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
        self.final_scores = {}
        self.content_metrics = Content_Metrics()

    def score(self, ref, hypo):
        # reset final score dictionary
        self.final_scores = {}
        for scorer, method in self.scorers:
            print("Metric: {}".format(method))
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    self.final_scores[m] = s
            else:
                self.final_scores[method] = score

    def compute_metrics(self, get_scores=True, live=False, **kwargs):
        pred = kwargs.pop('cands_ids', {})
        dataset = kwargs.pop('dataset', {})
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
        self.score(ref, hypo)
        precision, recall, f1, ndld = self.content_metrics(pred, dataset)
        self.final_scores['precision'] = 100.0*precision
        self.final_scores['recall'] = 100.0*recall
        self.final_scores['f1'] = 100.0*f1
        self.final_scores['ndld'] = 100.0*ndld

        for k, v in self.final_scores.items():
            print('[epoch-{}]{}:\t{}'.format(epoch, k, v))

        if get_scores:
            return self.final_scores

    def remove_dups(self, ids, tks):
        seq = [x for x,y in zip(ids, tks[0].split()) if y.isdigit()]
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def run_logger(self, writer, epoch, cat='valid_metrics'):
        rouge_l = self.final_scores['ROUGE_L']
        bleu_1 = self.final_scores['Bleu_1']
        bleu_2 = self.final_scores['Bleu_2']
        bleu_3 = self.final_scores['Bleu_3']
        bleu_4 = self.final_scores['Bleu_4']
        precision = self.final_scores['precision']
        recall = self.final_scores['recall']
        f1 =self. final_scores['f1']
        dis = self.final_scores['ndld']
        writer.add_scalar('{}/ROUGE_L'.format(cat), rouge_l, epoch)
        writer.add_scalar('{}/Bleu_1'.format(cat), bleu_1, epoch)
        writer.add_scalar('{}/Bleu_2'.format(cat), bleu_2, epoch)
        writer.add_scalar('{}/Bleu_3'.format(cat), bleu_3, epoch)
        writer.add_scalar('{}/Bleu_4'.format(cat), bleu_4, epoch)
        writer.add_scalar('{}/precision'.format(cat), precision, epoch)
        writer.add_scalar('{}/recall'.format(cat), recall, epoch)
        writer.add_scalar('{}/f1'.format(cat), f1, epoch)
        writer.add_scalar('{}/ndld'.format(cat), dis, epoch)

if __name__ == '__main__':
    cand = {'generated_description1': 'how are you', 'generated_description2': 'Hello how are you'}
    ref = {'generated_description1': ['what are you', 'where are you'],
           'generated_description2': ['Hello how are you', 'Hello how is your day']}
    print(normalized_damerau_levenshtein_distance('1234', '3150'))
    cands_ids = {1: [1, 2, 3, 4]}
    tgts_ids = {1: [3, 1, 5, 0]}
    x = Metrics()
    x.compute_metrics(get_scores=False, live=True, cand=cand, ref=ref, cands_ids=cands_ids, tgts_ids=tgts_ids)
