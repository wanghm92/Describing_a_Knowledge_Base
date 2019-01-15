import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from predictor import Predictor
from utils.loader import Table2text_seq
# from utils.loader_wikibio import Table2text_seq
from structure_generator.EncoderRNN import EncoderRNN
from structure_generator.DecoderRNN import DecoderRNN
from structure_generator.seq2seq import Seq2seq
from configurations import Config, ConfigTest
from eval import Evaluate
import random, os
from tensorboardX import SummaryWriter

# -------------------------------------------------------------------------------------------------- #
# ------------------------------------------- Args ------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #

parser = argparse.ArgumentParser(description='pointer generator model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='params.pkl',
                    help='path to save the final model')
parser.add_argument('--dataset', type=str, default='test', choices=['test', 'valid'],
                    help='type of dataset for prediction')
parser.add_argument('--mode', type=int, default=0, choices=[0, 1, 2, 3, 4],
                    help='train(0)/predict_individual(1)/predict_file(2)/compute score(3) or keep train (4)')
parser.add_argument('--type', type=int, default=0, choices=[0, 1],
                    help='person(0)/animal(1)')
parser.add_argument('--mask', action='store_true',
                    help='false(0)/true(1)')
parser.add_argument('--attn_type', type=str, default='concat', choices=['concat', 'dot'],
                    help='type of attention score calculation: concat, dot')
parser.add_argument('--attn_fuse', type=str, default='concat', choices=['concat', 'prod'],
                    help='type of attention score aggregation: sum, prod')
parser.add_argument('--attn_level', type=int, default=2, choices=[1, 2, 3],
                    help='levels of attention: 1(hidden only), 2(hidden+field), 3(hidden+word+field) hidden=rnn/emb')
parser.add_argument('--attn_src', type=str, default='emb', choices=['emb', 'rnn'],
                    help='encodings for attention layer: RNN hidden state(rnn) or word embeddings(emb)')
parser.add_argument('--field_self_att', action='store_true',
                    help='whether use field self-attention')
parser.add_argument('--use_cov_attn', action='store_true',
                    help='whether use coverage attention')
parser.add_argument('--use_cov_loss', action='store_true',
                    help='whether use coverage loss')
parser.add_argument('--field_concat_pos', action='store_true',
                    help='whether concat pos embeddings to field embeddings for attention calculation')
args = parser.parse_args()

# ------------------------------------- checking attn_src -------------------------------------- #
if args.attn_level == 3 and args.attn_src == 'emb':
    print(" *** WARNING *** args.attn_level == 3 and args.attn_src == 'emb', forcing attn_src to 'rnn'")
    args.attn_src = 'rnn'

# ------------------------------------- random seed and cuda -------------------------------------- #
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print(" *** WARNING *** CUDA device available, forcing to use")
        args.cuda = torch.cuda.is_available()
    else:
        torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

# --------------------------------------- save_file_dir ------------------------------------------- #
save_file_dir = os.path.dirname(args.save)
print("Models are going to be saved/loaded to/from: {}".format(save_file_dir))
if not os.path.exists(save_file_dir):
    print("save directory does not exist, mkdir ...")
    os.mkdir(save_file_dir)
    if not os.path.exists(os.path.join(save_file_dir, "evaluations")):
        os.mkdir(os.path.join(save_file_dir, "evaluations"))

# -------------------------------- Hyperparams and Tensorboard ------------------------------------ #
config = Config()
summary_dir = os.path.join(save_file_dir, "summary")
if not os.path.exists(summary_dir):
    os.mkdir(summary_dir)
writer = SummaryWriter(summary_dir)

# --------------------------------------- file suffix --------------------------------------------- #
if args.mask == 1:
    filepost = "_m"
else:
    filepost = ""

if args.type == 1:
    filepost += "_A.txt"
else:
    filepost += "_P.txt"

# -------------------------------------------------------------------------------------------------- #
# ------------------------------------ Training Functions ------------------------------------------ #
# -------------------------------------------------------------------------------------------------- #

def train_batch(dataset, batch_idx, model, teacher_forcing_ratio):
    batch_s, batch_o_s, batch_f, batch_pf, batch_pb, batch_t, batch_o_t, source_len, max_source_oov = \
        dataset.get_batch(batch_idx)

    losses = model(batch_s, batch_o_s, batch_f, batch_pf, batch_pb,
                   target=batch_t, target_id=batch_o_t,
                   input_lengths=source_len, max_source_oov=max_source_oov,
                   teacher_forcing_ratio=teacher_forcing_ratio)

    batch_loss = losses.mean()
    model.zero_grad()
    batch_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    return batch_loss.item(), len(source_len)


def train_epoches(t_dataset, v_dataset, model, n_epochs, teacher_forcing_ratio, load_epoch=0):
    eval_f = Evaluate()
    best_dev_bleu = 0.0
    best_dev_rouge = 0.0
    train_loader = t_dataset.corpus
    len_batch = len(train_loader)
    print("{} batches to be trained".format(len_batch))
    epoch_examples_total = t_dataset.len
    save_prefix = '.'.join(args.save.split('.')[:-1]) if args.mode == 4 else args.save
    print("Model saving prefix is: {}".format(save_prefix))

    for epoch in range(load_epoch + 1, n_epochs + load_epoch + 1):
        # --------------------------------------- train -------------------------------------------- #
        model.train(True)
        torch.set_grad_enabled(True)
        epoch_loss = 0

        batch_indices = list(range(len_batch))
        random.shuffle(batch_indices)

        for idx, batch_idx in enumerate(batch_indices):
            loss, num_examples = train_batch(t_dataset, batch_idx, model, teacher_forcing_ratio)
            epoch_loss += loss * num_examples
            sys.stdout.write('%d batches trained. current batch loss: %f\r' % (idx, loss))
            sys.stdout.flush()

        epoch_loss /= epoch_examples_total
        print("Finished epoch %d with average loss: %.4f" % (epoch, epoch_loss))
        writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)

        # --------------------------------------- inference -------------------------------------------- #
        predictor = Predictor(model, v_dataset.vocab, args.cuda)
        print("Start Evaluating ...")
        cand, ref = predictor.preeval_batch(v_dataset)
        print('Result:')
        print('ref: ', ref[1][0])
        print('cand: {}'.format(cand[1]))
        eval_file_out = "{}/evaluations/valid.epoch_{}.cand.live.txt".format(save_file_dir, load_epoch)
        with open(eval_file_out, 'w+') as fout:
            for c in range(len(cand)):
                fout.write("{}\n".format(cand[c+1]))

        # --------------------------------------- evaluation -------------------------------------------- #
        final_scores = eval_f.evaluate(live=True, cand=cand, ref=ref, epoch=epoch)
        rouge_l = final_scores['ROUGE_L']
        bleu_4 = final_scores['Bleu_4']
        writer.add_scalar('valid/ROUGE_L', rouge_l, epoch)
        writer.add_scalar('valid/Bleu_4', bleu_4, epoch)

        # ------------------------------------------ save ----------------------------------------------- #
        if bleu_4 >= best_dev_bleu and rouge_l >= best_dev_rouge:
            suffix = ".best_bleu_rouge"
            best_dev_bleu = bleu_4
            best_dev_rouge = rouge_l
        elif bleu_4 >= best_dev_bleu:
            suffix = ".best_bleu"
            best_dev_bleu = bleu_4
        elif rouge_l >= best_dev_rouge:
            suffix = ".best_rouge"
            best_dev_rouge = rouge_l
        else:
            suffix = ""
        if len(suffix) > 0:
            print("model at epoch #{} saved".format(epoch))
        else:
            print("[*** {} ***] model at epoch #{} saved".format(suffix, epoch))
        torch.save(model.state_dict(), "{}{}.{}".format(save_prefix, suffix, epoch))

        # epoch_score = 2*rouge_l*bleu_4/(rouge_l + bleu_4)

# -------------------------------------------------------------------------------------------------- #
# ------------------------------------------- Main ------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":

    # -------------------------------------------------------------------------------------------------- #
    # ------------------------------------- Reading Datasets ------------------------------------------- #
    # -------------------------------------------------------------------------------------------------- #
    print("Reading training data ...")
    t_dataset = Table2text_seq('train', type=args.type, USE_CUDA=args.cuda,
                               batch_size=config.batch_size, train_mode=args.mode)

    # -------------------------------------------------------------------------------------------------- #
    # -------------------------------------- Building Model -------------------------------------------- #
    # -------------------------------------------------------------------------------------------------- #
    print("Building Model ...")
    embedding = nn.Embedding(t_dataset.vocab.size, config.emsize, padding_idx=0)
    encoder = EncoderRNN(vocab_size=t_dataset.vocab.size, embedding=embedding, hidden_size=config.emsize,
                         pos_size=t_dataset.max_p, pemsize=config.pemsize, attn_src=args.attn_src,
                         input_dropout_p=config.dropout, dropout_p=config.dropout, n_layers=config.nlayers,
                         rnn_cell=config.cell, directions=config.directions,
                         variable_lengths=True, field_concat_pos=args.field_concat_pos)
    decoder = DecoderRNN(vocab_size=t_dataset.vocab.size, embedding=embedding, embed_size=config.emsize,
                         pemsize=config.pemsize, sos_id=3, eos_id=2, unk_id=1,
                         rnn_cell=config.cell, directions=config.directions,
                         attn_src=args.attn_src, attn_level=args.attn_level,
                         attn_type=args.attn_type, attn_fuse=args.attn_fuse,
                         use_cov_attn=args.use_cov_attn, use_cov_loss=args.use_cov_loss,
                         field_self_att=args.field_self_att, field_concat_pos=args.field_concat_pos, mask=args.mask,
                         use_cuda=args.cuda,
                         input_dropout_p=config.dropout, dropout_p=config.dropout, n_layers=config.nlayers)
    model = Seq2seq(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # --------------------------------------- train -------------------------------------------- #
    if args.mode == 0:
        try:
            print("number of training examples: %d" % t_dataset.len)
            print("Reading valid data ...")
            v_dataset = Table2text_seq('valid', type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
            print("start training...")
            train_epoches(t_dataset, v_dataset, model, config.epochs, teacher_forcing_ratio=1)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    # ------------------------------------ predict one ----------------------------------------- #
    elif args.mode == 1:
        model.load_state_dict(torch.load(args.save))
        print("model restored")
        dataset = Table2text_seq(args.dataset, type=args.type, USE_CUDA=args.cuda, batch_size=1)
        print("Read $-{}-$ data")
        predictor = Predictor(model, dataset.vocab, args.cuda)
        while True:
            seq_str = input("Type index from (%d to %d) to continue:\n" %(0, dataset.len - 1))
            i = int(seq_str)
            batch_s, batch_o_s, batch_f, batch_pf, batch_pb, sources, targets, fields, list_oovs, source_len \
                , max_source_oov, w2fs = dataset.get_batch(i)
            table = []
            for i in range(len(sources[0])):
                table.append(fields[0][i])
                table.append(":")
                table.append(sources[0][i])
            print("Table:")
            print(' '.join(table)+'\n')
            print("Refer: ")
            print(' '.join(targets[0])+'\n')
            outputs = predictor.predict(batch_s, batch_o_s, batch_f, batch_pf, batch_pb, max_source_oov
                                        , source_len, list_oovs[0], w2fs)
            print("Result: ")
            print(outputs)
            print('-'*120)

    # ------------------------------------ predict file ---------------------------------------- #
    elif args.mode == 2:
        model.load_state_dict(torch.load(args.save))
        load_epoch = int(args.save.split('.')[-1])
        print("model restored")
        dataset = Table2text_seq(args.dataset, type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
        print("Read $-{}-$ data")
        predictor = Predictor(model, dataset.vocab, args.cuda)
        print("number of test examples: %d" % dataset.len)
        print("Start Evaluating ...")
        lines = predictor.predict_file(dataset)

        print("Start writing")
        f_out = open("{}/evaluations/Output{}".format(save_file_dir, filepost), 'w')
        f_out.writelines(lines)
        f_out.close()

    # ----------------------------------- compute score ---------------------------------------- #
    elif args.mode == 3:
        model.load_state_dict(torch.load(args.save))
        load_epoch = int(args.save.split('.')[-1])
        print("model restored")
        dataset = Table2text_seq(args.dataset, type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
        print("Read $-{}-$ data")
        predictor = Predictor(model, dataset.vocab, args.cuda)
        print("number of test examples: %d" % dataset.len)
        print("Start Evaluating ...")
        cand, ref = predictor.preeval_batch(dataset)

        print('Result:')
        print('ref: ', ref[1][0])
        print('cand: {}'.format(cand[1]))
        cand_file_out = "{}/evaluations/{}.epoch_{}.cand.txt".format(save_file_dir, args.dataset, load_epoch)
        with open(cand_file_out, 'w+') as fout:
            for c in range(len(cand)):
                fout.write("{}\n".format(cand[c+1]))
        ref_file_out = "{}/evaluations/{}.ref.txt".format(save_file_dir, args.dataset)
        with open(ref_file_out, 'w+') as fout:
            for r in range(len(ref)):
                fout.write("{}\n".format(ref[r+1][0]))
        eval_f = Evaluate()
        final_scores = eval_f.evaluate(live=True, cand=cand, ref=ref, epoch=load_epoch)

    # ----------------------------------- resume train ----------------------------------------- #
    elif args.mode == 4:
        # load and keep training
        model.load_state_dict(torch.load(args.save))
        load_epoch = int(args.save.split('.')[-1])
        print("model restored from epoch-{}: {}".format(load_epoch, args.save))
        try:
            print("number of training examples: %d" % t_dataset.len)
            print("Reading valid data ...")
            v_dataset = Table2text_seq('valid', type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
            print("start training...")
            train_epoches(t_dataset, v_dataset, model, config.epochs, teacher_forcing_ratio=1, load_epoch=load_epoch)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        dataset = Table2text_seq(args.dataset, type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size)
        print("Read $-{}-$ data")
        predictor = Predictor(model, dataset.vocab, args.cuda)
        print("number of test examples: %d" % dataset.len)
        print("Start Evaluating ...")
        cand, ref = predictor.preeval_batch