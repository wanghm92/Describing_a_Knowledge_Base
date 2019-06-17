import io, os, sys
from collections import Counter
from tqdm import tqdm
DOMAIN = sys.argv[1]

DELIM = "￨"

filenames = ["inf_src_%s.txt"%DOMAIN, "src_%s.txt"%DOMAIN, "%s_content_plan_ids.txt"%DOMAIN, "%s.json"%DOMAIN,
             "roto-gold-%s.h5-tuples.txt", "tgt_%s.txt"%DOMAIN, "%s_content_plan_tks.txt"%DOMAIN, "%s-roto-ptrs.txt"%DOMAIN,
             "clean/tgt_%s.norm.mwe.txt"%DOMAIN, "clean/tgt_%s.norm.filter.mwe.txt"%DOMAIN,
             "clean/src_%s.norm.tk.txt"%DOMAIN, "clean/src_%s.norm.ext.txt"%DOMAIN,
             "clean/%s_content_plan_tks.txt"%DOMAIN, "clean/%s_content_plan_ids.txt"%DOMAIN, "clean/%s_ptrs.txt"%DOMAIN,
             "clean/tgt_%s.norm.filter.mwe.trim.txt"%DOMAIN, "src_%s.norm.tk.trim.txt"%DOMAIN]

in_path = os.path.join("/home/hongmin_wang/table2text_nlg/harvardnlp/data2text-plan-py/rotowire/", DOMAIN)

inf_src, src, cp_ids, js, gold_tps, \
tgt, cp_tks, ptrs, \
clean_tgt, clean_tgt_filter, \
clean_src, clean_src_ext, \
cp_out_tks, cp_out_ids, ptrs_out, \
clean_tgt_trim, clean_src_trim = [os.path.join(in_path, f) for f in filenames]

# TODO: remove NAs from src
import pickle

data = {}

def process(f, is_tgt=False, lower=False, name=''):
    cnt = 0
    ent_vals = []
    ent_nams = []
    record_types = []
    homeaways = []
    with io.open(f, 'r', encoding='utf-8') as fin:
        dataset = fin.read()
        if lower:
            dataset = dataset.lower()
        dataset = dataset.strip().split('\n')
        for sample in tqdm([x for x in dataset if len(x) > 0]):
            records = sample.strip().split()
            cnt += len(records)
            ent_vals.append([rcd.strip().split(DELIM)[0] for rcd in records])
            ent_nams.append([rcd.strip().split(DELIM)[1] for rcd in records])
            record_types.append([rcd.strip().split(DELIM)[2] for rcd in records])
            homeaways.append([rcd.strip().split(DELIM)[3] for rcd in records])

    print(len(ent_vals))
    print(len(ent_nams))
    print(len(record_types))
    print(len(ent_vals))
                
    if is_tgt:
        with io.open(cp_out_ids, 'r', encoding='utf-8') as fin:
            positions = [x.strip().split() for x in fin.read().strip().split('\n')]
            positions = [[int(x) for x in y] for y in positions]
        sequences = [zip(a, b, c, d, e) for a, b, c, d, e in zip(ent_nams, ent_vals, record_types, homeaways, positions)]
    else:
        sequences = [zip(a, b, c, d) for a, b, c, d in zip(ent_nams, ent_vals, record_types, homeaways)]
    print("cnt = {}".format(cnt))
        
#     print(list(sequences[0]))
    data[name] = sequences

process(clean_src_trim, name='source')
process(cp_out_tks, is_tgt=True, name='outlines')

with io.open(clean_tgt_trim, 'r', encoding='utf-8') as fin, io.open(ptrs_out, 'r', encoding='utf-8') as f_ptr:
    summary_tks = [x.strip().split() for x in fin.read().strip().split('\n')]
    # data['summaries'] = summary_tks
    # '''
    summary_ptr = [x.strip().split() for x in f_ptr.read().strip().split('\n')]
    print(len([x for x in summary_ptr if len(x) == 0]))
    print(len(summary_ptr))
    max_positions = []
    for ptrs in summary_ptr:
        pos = [int(x.split(',')[0]) for x in ptrs]
        if len(pos) > 0:
            max_positions.append(max(pos))
        else:
            max_positions.append(0)
    assert all([x < len(y) for x, y in zip(max_positions, summary_tks)])
    data['summaries'] = list(zip(summary_tks, summary_ptr))
    #'''

for k,v in data.items():
    print(len(v))

output = "/mnt/bhd/hongmin/table2text_nlg/datasets/dkb/rotowire_clean/{}_P.pkl".format(DOMAIN)
with io.open(output, 'wb+') as fout:
    pickle.dump(data, fout)
