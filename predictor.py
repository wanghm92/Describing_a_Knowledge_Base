import torch
import sys, os, math
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  #TkAgg
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

class Predictor(object):
    def __init__(self, model, vocab, USE_CUDA, decoder_type='pg', unk_gen='False', dataset_type=0):
        device = torch.device("cuda" if USE_CUDA else "cpu")
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.USE_CUDA = USE_CUDA
        self.decoder_type = decoder_type
        self.unk_gen = unk_gen
        self.dataset_type = dataset_type

    def predict(self, batch_s, batch_o_s, batch_f, batch_pf, batch_pb, max_source_oov, source_len, list_oovs, w2fs):
        torch.set_grad_enabled(False)
        decoded_outputs, lengths = self.model(batch_s, batch_o_s, batch_f, batch_pf, batch_pb,
                                              input_lenghts=source_len, max_source_oov=max_source_oov, w2fs=w2fs)
        length = lengths[0]
        output = []
        # print(decoded_outputs)
        for i in range(length):
            symbol = decoded_outputs[0][i].item()
            if symbol < self.vocab.size:
                output.append(self.vocab.idx2word[symbol])
            else:
                output.append(list_oovs[symbol-self.vocab.size])
        print(len(output))
        return ' '.join([i for i in output if i != '<PAD>' and i != '<EOS>' and i != '<SOS>'])

    def preeval_batch(self, dataset, fig=False, save_dir=None):
        torch.set_grad_enabled(False)
        refs = {}
        cands = {}
        if self.decoder_type == 'pt' and self.dataset_type == 3:
            cands_ids = {}
            tgts_ids = {}
            feats = {'fields': {}, 'rcds': {}, 'has': {}}
        else:
            cands_ids = None
            tgts_ids = None
            feats = {'fields': []}
        cands_with_pgens = {} if self.decoder_type == 'pg' else None
        cands_with_unks = {} if self.unk_gen else None
        srcs = {}
        i = 0
        pred_loss = 0
        figs_per_batch = 1
        token_count = 0
        total_batches = len(dataset.corpus)

        print("{} batches to be evaluated".format(total_batches))
        for batch_idx in tqdm(range(total_batches)):
            batch_s, batch_o_s, batch_f, batch_pf, batch_pb, batch_t, batch_o_t, source_len, max_source_oov, \
            w2fs, sources, targets, fields, list_oovs = dataset.get_batch(batch_idx)
            lab_t = batch_t[-1]

            everything = self.model(batch_s, batch_o_s, batch_f, batch_pf, batch_pb,
                                    w2fs=w2fs, input_lengths=source_len, max_source_oov=max_source_oov, fig=True)
            decouts, locations, lens, losses, p_gens, selfatt, attns = everything
            token_count += sum(lens)
            pred_loss += sum(losses)

            for j in range(len(lens)):
                i += 1
                srcs[i] = ' '.join(['_'.join(x.split()) for x in sources[j]])

                # NOTE: fields is a dictionary of all feats for ptr-net
                for k, v in fields.items():
                    feats[k][i] = ' '.join(v[j])

                refs[i] = [' '.join(targets[j])]
                out_seq_clean = []
                out_seq_unk = []
                if locations is not None:
                    out_seq_ids = locations[j].tolist()
                    out_seq_ids = out_seq_ids[:lens[j]-1]
                    cands_ids[i] = out_seq_ids
                    tgt_seq_ids = [x for x in lab_t[j].tolist() if x > 3]
                    tgts_ids[i] = tgt_seq_ids

                for k in range(lens[j]):
                    # get tokens and replace OOVs
                    symbol = decouts[j][k].item()
                    if symbol < self.vocab.size:
                        if symbol == 1 and self.unk_gen:
                            replacement = sources[j][attns[j][k].argmax()]
                            out_seq_clean.append(replacement)
                        else:
                            out_seq_clean.append(self.vocab.idx2word[symbol])
                        out_seq_unk.append(self.vocab.idx2word[symbol])
                    else:
                        oov = list_oovs[j][symbol-self.vocab.size]
                        out_seq_clean.append(oov)
                        out_seq_unk.append(oov)

                pgen = p_gens[j] if p_gens is not None else None
                out, out_unk, out_with_gens = self.post_process(out_seq_clean, out_seq_unk, pgen)

                cands[i] = ' '.join(out)
                # if locations is not None:
                #     cands_ids[i] = list(zip(out_seq_ids, out))

                if self.decoder_type == 'pg':
                    cands_with_pgens[i] = ' '.join(out_with_gens)
                if self.unk_gen:
                    cands_with_unks[i] = ' '.join(out_unk)

                if fig and len(out) > 0 and j < figs_per_batch:
                    fmatrix = selfatt[j] if selfatt is not None else None
                    self.make_figure(lens[j], out, fmatrix, attns[j], batch_pf[j], sources[j], batch_idx+j, save_dir)

        others = (cands_with_unks, cands_with_pgens, cands_ids, tgts_ids, srcs, feats)

        return cands, refs, math.exp(pred_loss/token_count), others

    def post_process(self, sentence, sentence_unk, pgen=None):
        try:
            eos = sentence.index('<EOS>')
            sentence = sentence[:eos]
            sentence_unk = sentence_unk[:eos]
            if pgen is not None:
                pgen = pgen[:eos]
        except ValueError:
            pass

        if pgen is None:
            sentence_trim = [x for x in sentence if x != '<PAD>' and x != '<EOS>' and x != '<SOS>']
            sentence_unk_trim = [x for x in sentence_unk if x != '<PAD>' and x != '<EOS>' and x != '<SOS>']
            return sentence_trim, sentence_unk_trim, []
        else:
            token_pgens = [(x, y, z.item()) for x, y, z in zip(sentence, sentence_unk, pgen)
                           if x != '<PAD>' and x != '<EOS>' and x != '<SOS>']
            if len(token_pgens) > 0:
                out, out_unk, pgens_filtered = zip(*token_pgens)
                pgens_filtered = ["_%.3f"%x if x < 0.7 else '' for x in pgens_filtered]
                out_with_gens = ["{}{}".format('_'.join(x.split()), y) for x, y in zip(out, pgens_filtered)]
                return list(out), list(out_unk), out_with_gens
            else:
                return [], [], []

    def showAttention(self, input_words, output_words, attentions, attn='self', xlabel="", ylabel="", idx=0, dir=''):
        # Set up figure with colorbar
        plt.rcParams.update({'font.size': 18})
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap=plt.cm.Blues)
        if attn == 'self':
            fig.colorbar(cax, pad=0.03)
        else:
            fig.colorbar(cax, shrink=0.7, pad=0.03)

        # Set up axes
        ax.set_xticklabels([''] + input_words, rotation=90)
        ax.set_yticklabels([''] + output_words)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        fig.savefig(os.path.join(dir, "attention_figures/{}.{}.png".format(attn, idx)), bbox_inches='tight')
        plt.close(fig)

    def make_figure(self, length, out, fmatrix, attns, batch_pf, src_words, idx, savedir):
        pos = [str(i) for i in batch_pf.cpu().tolist() if i >0]
        combine = []
        for j in range(len(pos)):
            combine.append(src_words[j] + " : " + pos[j])

        if fmatrix is not None:
            self.showAttention(pos, combine, fmatrix.cpu(),
                               attn='self', xlabel='Table Position', ylabel='Table Position', idx=idx, dir=savedir)
        self.showAttention(combine, out, attns[:len(out)].cpu(),
                           attn='attn', xlabel='Structured KB', ylabel='Text Output', idx=idx, dir=savedir)
