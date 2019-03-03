import sys
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from utils.text2num import text2num

class Content_Metrics(object):

    def __init__(self):

        full_names = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
         'Chicago Bulls', 'Cleveland Cavaliers', 'Detroit Pistons', 'Indiana Pacers',
         'Miami Heat', 'Milwaukee Bucks', 'New York Knicks', 'Orlando Magic',
         'Philadelphia 76ers', 'Toronto Raptors', 'Washington Wizards', 'Dallas Mavericks',
         'Denver Nuggets', 'Golden State Warriors', 'Houston Rockets', 'Los Angeles Clippers',
         'Los Angeles Lakers', 'Memphis Grizzlies', 'Minnesota Timberwolves', 'New Orleans Pelicans',
         'Oklahoma City Thunder', 'Phoenix Suns', 'Portland Trail Blazers', 'Sacramento Kings',
         'San Antonio Spurs', 'Utah Jazz']

        self.full_names = [x.lower() for x in full_names]

        self.cities, self.teams = set(), set()
        self.ec = {} # equivalence classes
        for team in self.full_names:
            pieces = team.split()
            if len(pieces) == 2:
                self.ec[team] = [pieces[0], pieces[1]]
                self.cities.add(pieces[0])
                self.teams.add(pieces[1])
            elif pieces[0] == "portland": # only 2-word team
                self.ec[team] = [pieces[0], " ".join(pieces[1:])]
                self.cities.add(pieces[0])
                self.teams.add(" ".join(pieces[1:]))
            else: # must be a 2-word City
                self.ec[team] = [" ".join(pieces[:2]), pieces[2]]
                self.cities.add(" ".join(pieces[:2]))
                self.teams.add(pieces[2])

    def __call__(self, pred_ids, dataset):
        gold_triples = dataset.gold_triples
        pred_triples = dataset._get_pred_triples(pred_ids)
        try:
            assert len(gold_triples) == len(pred_triples)
        except AssertionError:
            sys.exit("len(gold_triples) = {} but len(pred_triples) = {}".format(len(gold_triples), len(pred_triples)))

        precision, recall, f1 = self.f1(pred_triples, gold_triples)
        avg_score = self.dld(pred_triples, gold_triples)
        return precision, recall, f1, avg_score

    def dedup(self, triplist):
        """inefficient"""
        dups = set()
        for i in range(1, len(triplist)):
            for j in range(i):
                if self._record_match(triplist[i], triplist[j]):
                    dups.add(i)
                    break
        return [thing for i, thing in enumerate(triplist) if i not in dups]

    def _same_ent(self, e1, e2):
        if e1 in self.cities or e1 in self.teams or e2 in self.cities or e2 in self.teams:
            return e1 == e2 or any((e1 in fn and e2 in fn for fn in self.full_names))
        else:
            return e1 in e2 or e2 in e1

    def _int_value(self, input):
        a_number = False
        try:
            value = int(input)
            a_number = True
        except ValueError:
            pass

        if not a_number:
            value = text2num(input)
        return value

    def _record_match(self, t1, t2):
        return self._int_value(t1[1]) == self._int_value(t2[1]) \
               and t1[2] == t2[2] \
               and self._same_ent(t1[0], t2[0])

    def f1(self, pred_triples, gold_triples):

        total_tp, total_predicted, total_gold = 0, 0, 0
        for i, triplist in enumerate(pred_triples):
            tp = sum((1 for j in range(len(triplist))
                        if any(self._record_match(triplist[j], gold_triples[i][k])
                               for k in range(len(gold_triples[i])))))
            total_tp += tp
            total_predicted += len(triplist)
            total_gold += len(gold_triples[i])
        precision = float(total_tp)/total_predicted
        recall = float(total_tp)/total_gold
        f1 = 2*precision*recall/(precision+recall)
        # print("totals:", total_tp, total_predicted, total_gold)
        # print("precision:", precision, "recall:", recall)
        return precision, recall, f1

    def _norm_dld(self, l1, l2):
        ascii_start = 0
        # make a string for l1
        # all triples are unique...
        s1 = ''.join((chr(ascii_start+i) for i in range(len(l1))))
        s2 = ''
        next_char = ascii_start + len(s1)
        for j in range(len(l2)):
            found = None
            #next_char = chr(ascii_start+len(s1)+j)
            for k in range(len(l1)):
                if self._record_match(l2[j], l1[k]):
                    found = s1[k]
                    #next_char = s1[k]
                    break
            if found is None:
                s2 += chr(next_char)
                next_char += 1
                # assert next_char <= 128
                if next_char > 128:
                    print("WARNING: Max chars reached")
                    break
            else:
                s2 += found
        # return 1- , since this thing gives 0 to perfect matches etc
        return 1.0-normalized_damerau_levenshtein_distance(s1, s2)

    def dld(self, pred_triples, gold_triples):

        assert len(gold_triples) == len(pred_triples)
        total_score = 0
        for i, triplist in enumerate(pred_triples):
            total_score += self._norm_dld(triplist, gold_triples[i])
        avg_score = float(total_score)/len(pred_triples)
        return avg_score