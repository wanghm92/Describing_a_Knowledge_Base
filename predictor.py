import torch
import sys, os
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  #TkAgg
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

class Predictor(object):
    def __init__(self, model, vocab, USE_CUDA):
        device = torch.device("cuda" if USE_CUDA else "cpu")
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.USE_CUDA = USE_CUDA

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

    def preeval_batch(self, dataset, fig=False, save_file_dir=None):
        torch.set_grad_enabled(False)
        refs = {}
        cands = {}
        cands_with_pgens = {}
        srcs = {}
        fds = {}
        i = 0
        eval_loss = 0
        figs_per_batch = 1
        total_batches = len(dataset.corpus)
        print("{} batches to be evaluated".format(total_batches))
        for batch_idx in tqdm(range(total_batches)):
            batch_s, batch_o_s, batch_f, batch_pf, batch_pb, sources, targets, fields, list_oovs, source_len, \
                max_source_oov, w2fs = dataset.get_batch(batch_idx)
            decouts, lens, losses, p_gens, selfatt, attns = self.model(batch_s, batch_o_s, batch_f, batch_pf, batch_pb,
                                                                       w2fs=w2fs, input_lengths=source_len,
                                                                       max_source_oov=max_source_oov, fig=True)
            eval_loss += sum(losses)/len(losses)
            for j in range(len(lens)):
                i += 1
                srcs[i] = ' '.join(['_'.join(x.split()) for x in sources[j]])
                fds[i] = ' '.join(fields[j])
                refs[i] = [' '.join(targets[j])]
                out_seq = []
                for k in range(lens[j]):
                    # get tokens and replace OOVs
                    symbol = decouts[j][k].item()
                    if symbol < self.vocab.size:
                        out_seq.append(self.vocab.idx2word[symbol])
                    else:
                        out_seq.append(list_oovs[j][symbol-self.vocab.size])
                out, out_with_gens = self.post_process(out_seq, p_gens[j])
                if fig and len(out) > 0 and j < figs_per_batch:
                    self.make_figure(lens[j], out, selfatt[j], attns[j], batch_pf[j], sources[j], batch_idx+j, save_file_dir)
                cands[i] = ' '.join(out)
                cands_with_pgens[i] = ' '.join(out_with_gens)

        return cands, refs, eval_loss/total_batches, (cands_with_pgens, srcs, fds)

    def post_process(self, sentence, p_gens=None):
        try:
            eos = sentence.index('<EOS>')
            sentence = sentence[:eos]
            if p_gens is not None:
                p_gens = p_gens[:eos]
        except ValueError:
            pass

        if p_gens is None:
            return ' '.join([x for x in sentence if x != '<PAD>' and x != '<EOS>' and x != '<SOS>']), None
        else:
            token_pgens = [(x, y.item()) for x, y in zip(sentence, p_gens) if x != '<PAD>' and x != '<EOS>' and x != '<SOS>']
            if len(token_pgens) > 0:
                out, pgens_filtered = zip(*token_pgens)
                pgens_filtered = ["_%.3f"%x if x < 0.7 else '' for x in pgens_filtered]
                out_with_gens = ["{}{}".format('_'.join(x.split()), y) for x, y in zip(out, pgens_filtered)]
                return list(out), out_with_gens
            else:
                return [], []

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

    def make_figure(self, length, out, selfatt, attns, batch_pf, src_words, idx, savedir):
        pos = [str(i) for i in batch_pf.cpu().tolist() if i >0]
        combine = []
        for j in range(len(pos)):
            combine.append(src_words[j] + " : " + pos[j])

        if selfatt is not None:
            self.showAttention(pos, combine, selfatt.cpu(),
                               attn='self', xlabel='Table Position', ylabel='Table Position', idx=idx, dir=savedir)
        self.showAttention(combine, out, attns[:len(out)].cpu(),
                           attn='attn', xlabel='Structured KB', ylabel='Text Output', idx=idx, dir=savedir)
