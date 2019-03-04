import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from predictor import Predictor
from structure_generator.EncoderRNN import EncoderRNN
from structure_generator.DecoderRNN import DecoderRNN
from structure_generator.seq2seq import Seq2seq
from configurations import Config, ConfigSmall, ConfigTest, ConfigWikibio, ConfigRotowire
from metrics import Metrics
from validator import Validator
import random, os, pprint, logging, time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

DELIM = u"￨"
program = os.path.basename(sys.argv[0])
L = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
L.info("Running %s" % ' '.join(sys.argv))

# -------------------------------------------------------------------------------------------------- #
# ------------------------------------------- Args ------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #

parser = argparse.ArgumentParser(description='pointer generator model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--dec_type', type=str, default='pg', choices=['pg', 'pt', 'seq'],
                    help='decoder model type pg(pointer-generator)/pt(pointer-net)(WIP)/seq(seq2seq)')
parser.add_argument('--ptr_input', type=str, default='emb', choices=['emb', 'hid'],
                    help='input to pointer-network emb: normal word+feat/hidden: memory bank hidden vectors')
parser.add_argument('--dec_feat_merge', type=str, default='mlp', choices=['cat', 'mlp'],
                    help='merge input embeddings for decoder')
parser.add_argument('--enc_type', type=str, default='rnn', choices=['rnn', 'fc', 'trans'],
                    help='encoder model type')
parser.add_argument('--save', type=str, default='params.pkl',
                    help='path to save the final model')
parser.add_argument('--dataset', type=str, default='test', choices=['test', 'valid'],
                    help='type of dataset for prediction')
parser.add_argument('--mode', type=int, default=0, choices=[0, 1, 2],
                    help='train(0)/resume(1)/evaluation(2)')
parser.add_argument('--type', type=int, default=0, choices=[0, 1, 2, 3],
                    help='person(0)/animal(1)/wikibio(2)/rotowire(3)')
parser.add_argument('--mask', action='store_true',
                    help='false(0)/true(1)')
parser.add_argument('--batch', type=int, default='64',
                    help='batch size')
parser.add_argument('--max_len', type=int, default='100',
                    help='max_len')

parser.add_argument('--attn_type', type=str, default='cat', choices=['cat', 'dot'],
                    help='type of attention score calculation: cat, dot')
parser.add_argument('--attn_fuse', type=str, default='cat', choices=['cat', 'prod', 'no'],
                    help='type of attention score aggregation: cat, prod, no')
parser.add_argument('--attn_level', type=int, default=2, choices=[1, 2, 3],
                    help='levels of attention: 1(hidden only), 2(hidden+field), 3(hidden+word+field) hidden=rnn/emb')
parser.add_argument('--attn_src', type=str, default='emb', choices=['emb', 'rnn'],
                    help='encodings for attention layer: RNN hidden state(rnn) or word embeddings(emb)')

parser.add_argument('--use_cov_attn', action='store_true',
                    help='whether use coverage attention')
parser.add_argument('--use_cov_loss', action='store_true',
                    help='whether use coverage loss')
parser.add_argument('--cov_in_pgen', action='store_true',
                    help='whether use coverage in calculating p_gen')

parser.add_argument('--field_self_att', action='store_true',
                    help='whether use field self-attention')
parser.add_argument('--field_cat_pos', action='store_true',
                    help='whether cat pos embeddings to field embeddings for attention calculation')
parser.add_argument('--field_context', action='store_false',
                    help='whether pass context vector of field embeddings to output layer')

parser.add_argument('--pt_dec_feat', action='store_true',
                    help='whether to cat features for ptr-net decoder')

parser.add_argument('--context_mlp', action='store_true',
                    help='MLP layer on context vectors before output layer')

parser.add_argument('--shuffle', action='store_true',
                    help='whether to shuffle the batches during each epoch')

parser.add_argument('--fig', action='store_true',
                    help='generate attention visualization figures for evaluation')
parser.add_argument('--verbose', action='store_true',
                    help='print sample outputs')
parser.add_argument('--xavier', action='store_false',
                    help='xavier initialization')

args = parser.parse_args()

# ------------------------------------- checking attn_src -------------------------------------- #
if args.attn_level != 2 and args.attn_src == 'emb':
    L.info(" *** WARNING *** args.attn_level != 2 and args.attn_src == 'emb', forcing attn_src to 'rnn'")
    args.attn_src = 'rnn'

# ------------------------------------- random seed and cuda -------------------------------------- #
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        L.info(" *** WARNING *** CUDA device available, forcing to use")
        args.cuda = torch.cuda.is_available()
    else:
        torch.cuda.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

# --------------------------------------- save_file_dir ------------------------------------------- #
save_file_dir = os.path.dirname(args.save)
L.info("Models are going to be saved/loaded to/from: {}".format(save_file_dir))
if not os.path.exists(save_file_dir):
    L.info("save directory does not exist, mkdir ...")
    os.mkdir(save_file_dir)
    if not os.path.exists(os.path.join(save_file_dir, "evaluations")):
        os.mkdir(os.path.join(save_file_dir, "evaluations"))
    if not os.path.exists(os.path.join(save_file_dir, "attention_figures")):
        os.mkdir(os.path.join(save_file_dir, "attention_figures"))

# -------------------------------- Hyperparams and Tensorboard ------------------------------------ #
if args.type == 2:
    config = ConfigWikibio()
    # from utils.loader_wikibio import Table2text_seq
    from utils.loader_wikibio_pt import Table2text_seq
elif args.type == 3:
    config = ConfigRotowire()
    args.field_cat_pos = True
    from utils.loader_rotowire_pt import Table2text_seq
else:
    config = Config()
    from utils.loader import Table2text_seq

# config = ConfigTest()
print("\n***config: ")
pprint.pprint(vars(config), indent=2)

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

print("\n***args: ")
pprint.pprint(vars(args), indent=2)

# -------------------------------------------------------------------------------------------------- #
# ------------------------------------ Training Functions ------------------------------------------ #
# -------------------------------------------------------------------------------------------------- #

def train_batch(dataset, batch_idx, model, teacher_forcing_ratio):
    """ Train with one batch """
    batch_s, batch_o_s, batch_f, batch_pf, batch_pb, batch_t, batch_o_t, source_len, max_source_oov = \
        dataset.get_batch(batch_idx)

    losses = model(batch_s, batch_o_s, batch_f, batch_pf, batch_pb,
                   target=batch_t, target_id=batch_o_t,
                   input_lengths=source_len, max_source_oov=max_source_oov,
                   teacher_forcing_ratio=teacher_forcing_ratio)

    batch_loss = losses.mean()
    model.zero_grad()
    batch_loss.backward()
    total_norm = 0.0
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    return batch_loss.item(), len(source_len), total_norm

def train(trainsets, v_dataset, model, n_epochs, teacher_forcing_ratio, load_epoch=0):
    """
    Train epochs, evaluate and inference on valid set
    Evaluation starts from epoch 0: random initialization
    """
    t_dataset, t4e_dataset = trainsets
    best_dev_bleu = 0.0
    best_dev_rouge = 0.0
    train_loader = t_dataset.corpus
    len_batch = len(train_loader)
    epoch_examples_total = t_dataset.len
    save_prefix = '.'.join(args.save.split('.')[:-1]) if args.mode == 4 else args.save
    L.info("Model saving prefix is: {}".format(save_prefix))

    epoch = load_epoch
    while True:
        # ------------------------------------------------------------------------------------------ #
        # ----------------------------------- Eval on Valid set ------------------------------------ #
        # ------------------------------------------------------------------------------------------ #
        L.info("Validation Epoch - {}".format(epoch))
        valid_f = Validator(model=model, v_dataset=v_dataset, use_cuda=args.cuda, tfr=teacher_forcing_ratio)
        valid_loss = valid_f.valid()
        writer.add_scalar('loss/valid_loss', valid_loss, epoch)

        # ------------------------------------------------------------------------------------------ #
        # --------------------------------- Inference on Valid set --------------------------------- #
        # ------------------------------------------------------------------------------------------ #
        L.info("Inference Epoch - {}".format(epoch))
        predictor = Predictor(model, v_dataset.vocab, args.cuda,
                              decoder_type=args.dec_type, unk_gen=config.unk_gen, dataset_type=args.type)
        cand, ref, perplexity, others = predictor.preeval_batch(v_dataset)
        cands_with_unks, cands_with_pgens, cands_ids, tgts_ids, srcs, feats = others
        if epoch > 0:
            writer.add_scalar('perplexity/valid', perplexity, epoch)

        L.info('Result:')
        L.info('valid_loss: {}'.format(valid_loss))
        L.info('perplexity: {}'.format(perplexity))
        if args.verbose:
            L.info('\nsource[1]: {}'.format(srcs[1]))
            for k, v in feats.items():
                L.info('\n{}[1]: {}'.format(k, v[1]))
            if config.unk_gen:
                print('\ncands_with_unks[1]: {}'.format(cands_with_unks[1]))
            if cands_with_pgens is not None:
                print('\ncands_with_pgens[1]: {}'.format(cands_with_pgens[1]))
            if cands_ids is not None:
                print('\ncands_ids[1]: {}'.format(cands_ids[1]))
            if tgts_ids is not None:
                print('\ntgts_ids[1]: {}'.format(tgts_ids[1]))
            print('\nref[1]: {}'.format(ref[1][0]))
            print('\ncand[1]: {}'.format(cand[1]))

        eval_file_out = "{}/evaluations/valid.epoch_{}.cand.live.txt".format(save_file_dir, epoch)
        with open(eval_file_out, 'w+') as fout:
            for c in range(len(cand)):
                fout.write("{}\n".format(cand[c+1]))
        if config.unk_gen:
            eval_file_out_unk = "{}/evaluations/valid.epoch_{}.cand.unk.txt".format(save_file_dir, epoch)
            with open(eval_file_out_unk, 'w+') as fout:
                for c in range(len(cands_with_unks)):
                    fout.write("{}\n".format(cands_with_unks[c + 1]))
        if cands_with_pgens is not None:
            eval_file_out_pgen = "{}/evaluations/valid.epoch_{}.cand.pgen.txt".format(save_file_dir, epoch)
            with open(eval_file_out_pgen, 'w+') as fout:
                for c in range(len(cands_with_pgens)):
                    fout.write("{}\n".format(cands_with_pgens[c + 1]))
        if cands_ids is not None:
            eval_file_out_ids = "{}/evaluations/valid.epoch_{}.cand.ids.txt".format(save_file_dir, epoch)
            cands_ids_original = [cands_ids[i+1] for i in v_dataset.sort_indices]
            with open(eval_file_out_ids, 'w+') as fout:
                for c in range(len(cands_ids_original)):
                    fout.write("{}\n".format(" ".join([str(x) for x in cands_ids_original[c]])))

        # ------------------------------------------------------------------------------------------ #
        # --------------------------------- Metrics on Valid set ----------------------------------- #
        # ------------------------------------------------------------------------------------------ #
        valid_scores = metrics.compute_metrics(live=True, cand=cand, ref=ref, epoch=epoch,
                                               cands_ids=cands_ids_original, dataset=v_dataset, tgts_ids=tgts_ids)
        metrics.run_logger(writer=writer, epoch=epoch)

        # ------------------------------------------------------------------------------------------ #
        # ---------------------------- Eval and Metrics on Training set ---------------------------- #
        # ------------------------------------------------------------------------------------------ #
        cand, ref, perplexity, others = predictor.preeval_batch(t4e_dataset)
        _, _, train_cands_ids, train_tgts_ids, _, _ = others
        if epoch > 0:
            writer.add_scalar('perplexity/train', perplexity, epoch)
        if train_cands_ids is not None:
            train_cands_ids_original = [train_cands_ids[i+1] for i in t4e_dataset.sort_indices]
        _ = metrics.compute_metrics(live=True, cand=cand, ref=ref, epoch=epoch,
                                    cands_ids=train_cands_ids_original, dataset=t4e_dataset, tgts_ids=train_tgts_ids)
        metrics.run_logger(writer=writer, epoch=epoch, cat='train_metrics')

        # ------------------------------------------------------------------------------------------ #
        # ------------------------------------------ save ------------------------------------------ #
        # ------------------------------------------------------------------------------------------ #
        rouge_l = valid_scores['ROUGE_L']
        bleu_4 = valid_scores['Bleu_4']
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
        if len(suffix) > 0 and epoch > 1:
            L.info("model at epoch #{} saved".format(epoch))
            torch.save(model.state_dict(), "{}{}.{}".format(save_prefix, suffix, epoch))

        # ------------------------------------------------------------------------------------------ #
        # --------------------------------------- train -------------------------------------------- #
        # ------------------------------------------------------------------------------------------ #
        epoch += 1
        if epoch > n_epochs + load_epoch:
            break

        L.info("Training Epoch - {}".format(epoch))
        L.info("{} batches to be trained".format(len_batch))

        model.train(True)
        torch.set_grad_enabled(True)
        epoch_loss = 0.0

        batch_indices = list(range(len_batch))
        batch_indices.reverse()  # start from the short ones
        if args.shuffle:
            batch_indices = np.random.permutation(batch_indices)

        start_time = time.time()
        for idx, batch_idx in enumerate(batch_indices):
            loss, num_examples, total_norm = train_batch(t_dataset, batch_idx, model, teacher_forcing_ratio)
            epoch_loss += loss * num_examples
            if idx%1 == 0:
                end_time = time.time()
                sys.stdout.write('%d batches trained. current batch loss: %f [%.3fs]\r' % (idx, loss, end_time-start_time))
                sys.stdout.flush()
            writer.add_scalar('batch/loss', loss, (epoch-1)*len_batch+idx+1)
            writer.add_scalar('batch/grad_norm', total_norm, (epoch-1)*len_batch+idx+1)

        epoch_loss /= epoch_examples_total
        L.info("Finished epoch %d with average loss: %.4f" % (epoch, epoch_loss))
        writer.add_scalar('loss/train_loss', epoch_loss, epoch)
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(valid_loss)
            else:
                scheduler.step()

# -------------------------------------------------------------------------------------------------- #
# ------------------------------------------- Main ------------------------------------------------- #
# -------------------------------------------------------------------------------------------------- #
# noinspection PyUnboundLocalVariable
if __name__ == "__main__":
    metrics = Metrics()
    # -------------------------------------------------------------------------------------------------- #
    # ------------------------------------- Reading Datasets ------------------------------------------- #
    # -------------------------------------------------------------------------------------------------- #
    L.info("Reading training data ...")
    t_dataset = Table2text_seq('train', type=args.type, USE_CUDA=args.cuda, batch_size=config.batch_size,
                               train_mode=args.mode, dec_type=args.dec_type)
    t4e_dataset = Table2text_seq('train4eval', type=args.type, USE_CUDA=args.cuda, batch_size=config.valid_batch,
                               train_mode=args.mode, dec_type=args.dec_type)
    # -------------------------------------------------------------------------------------------------- #
    # -------------------------------------- Building Model -------------------------------------------- #
    # -------------------------------------------------------------------------------------------------- #
    L.info("Building Model ...")
    embedding = nn.Embedding(t_dataset.vocab.size, config.emsize, padding_idx=0)

    if args.type == 3:
        assert hasattr(t_dataset.vocab, 'field_vocab_size')
        assert hasattr(t_dataset.vocab, 'rcd_vocab_size')
        assert hasattr(t_dataset.vocab, 'ha_vocab_size')
        hidden_size = config.hdsize
        fd_size = config.fdsize
        rcd_size = config.rcdsize
        ha_size = config.hasize
        field_embedding = nn.Embedding(t_dataset.vocab.field_vocab_size, fd_size, padding_idx=0)
        rcd_embedding = nn.Embedding(t_dataset.vocab.rcd_vocab_size, rcd_size, padding_idx=0)
        ha_embedding = nn.Embedding(t_dataset.vocab.ha_vocab_size, ha_size, padding_idx=0)
        posit_size = rcd_size + ha_size
        pos_embedding = (rcd_embedding, ha_embedding)
    else:
        pos_vocab_size = t_dataset.max_p
        pos_embedding = nn.Embedding(pos_vocab_size, config.pemsize, padding_idx=0)
        posit_size = config.pemsize * 2

        if args.type == 2:
            assert hasattr(t_dataset.vocab, 'field_vocab_size')
            field_embedding = nn.Embedding(t_dataset.vocab.field_vocab_size, config.fdsize, padding_idx=0)
            hidden_size = config.hdsize
            fd_size = config.fdsize
        else:
            field_embedding = None
            hidden_size = config.emsize
            fd_size = config.emsize

    encoder = EncoderRNN(vocab_size=t_dataset.vocab.size, embedding=embedding,
                         embed_size=config.emsize, fdsize=fd_size, hidden_size=hidden_size,
                         posit_size=posit_size, dec_size=hidden_size, attn_size=config.attn_size,
                         attn_src=args.attn_src,
                         dropout_p=config.dropout, n_layers=config.nlayers,
                         rnn_cell=config.cell, directions=config.directions,
                         variable_lengths=True, field_cat_pos=args.field_cat_pos,
                         field_embedding=field_embedding, pos_embedding=pos_embedding,
                         dataset_type=args.type, enc_type=args.enc_type)
    decoder = DecoderRNN(dec_type=args.dec_type, ptr_input=args.ptr_input, dec_feat_merge=args.dec_feat_merge,
                         vocab_size=t_dataset.vocab.size, embedding=embedding,
                         embed_size=config.emsize, hidden_size=hidden_size, fdsize=fd_size, posit_size=posit_size,
                         sos_id=3, eos_id=2, unk_id=1,
                         rnn_cell=config.cell, directions=config.directions,
                         attn_src=args.attn_src, attn_level=args.attn_level,
                         attn_type=args.attn_type, attn_fuse=args.attn_fuse,
                         pt_dec_feat=args.pt_dec_feat,
                         use_cov_attn=args.use_cov_attn, use_cov_loss=args.use_cov_loss, cov_in_pgen=args.cov_in_pgen,
                         field_self_att=args.field_self_att, field_cat_pos=args.field_cat_pos,
                         field_context=args.field_context, context_mlp=args.context_mlp,
                         mask=args.mask, use_cuda=args.cuda, unk_gen=config.unk_gen,
                         max_len=config.max_len, min_len=config.min_len,
                         dropout_p=config.dropout, n_layers=config.nlayers,
                         field_embedding=field_embedding, pos_embedding=pos_embedding, dataset_type=args.type)

    model = Seq2seq(encoder, decoder).to(device)
    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config.decay_rate) if config.decay_rate < 1 else None
        # milestones = list(range(config.decay_start, config.epochs))
        # scheduler = MultiStepLR(optimizer, milestones, gamma=config.decay_rate) if config.decay_rate < 1 else None
        # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)
        # TODO: use plateau scheduler
        scheduler = ExponentialLR(optimizer, gamma=config.decay_rate) if config.decay_rate < 1 else None
    elif config.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config.lr, lr_decay=config.decay_rate,
                                  initial_accumulator_value=0.1)
        scheduler = None
    else:
        raise ValueError("{} optimizer not supported".format(args.optim))

    # -------------------------------------------------------------------------------------------------- #
    # -------------------------------------- Model parameters ------------------------------------------ #
    # -------------------------------------------------------------------------------------------------- #
    L.info("Model parameters: ")
    params_dict = {}
    if args.xavier:
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'rnn' in name or 'V' in name or 'embedding' in name:
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                        params_dict["[Constant-0][{}] {}".format(param.dtype, name)] = param.size()
                        # print("Constant(0): {}".format(name))
                    else:
                        nn.init.xavier_uniform_(param)
                        params_dict["[Xavier][{}] {}".format(param.dtype, name)] = param.size()
                else:
                    try:
                        nn.init.xavier_uniform_(param)
                        params_dict["[Xavier][{}] {}".format(param.dtype, name)] = param.size()
                    except:
                        if param.size()[0] == 1:
                            nn.init.constant_(param, 0.0)
                            params_dict["[Constant-0][{}] {}".format(param.dtype, name)] = param.size()
                        else:
                            params_dict["[Uniform: 1/dim*0.5][{}] {}".format(param.dtype, name)] = param.size()

        pprint.pprint(params_dict, indent=2)
    else:
        pprint.pprint(model.named_parameters(), indent=2)
    # ------------------------------------------------------------------------------------------ #
    # --------------------------------------- train -------------------------------------------- #
    # ------------------------------------------------------------------------------------------ #
    if args.mode == 0:
        try:
            L.info("number of training examples: %d" % t_dataset.len)
            L.info("Reading valid data ...")
            v_dataset = Table2text_seq('valid', type=args.type, USE_CUDA=args.cuda,
                                       batch_size=config.valid_batch, dec_type=args.dec_type)

            L.info("start training...")
            trainsets = (t_dataset, t4e_dataset)
            train(trainsets, v_dataset, model, config.epochs, teacher_forcing_ratio=1)
            writer.close()
        except KeyboardInterrupt:
            L.info('-' * 89)
            writer.close()
            torch.save(model.state_dict(), "{}/model_before_kill.pkl".format(save_file_dir))
            L.info("Model saved at: {}/model_before_kill.pkl".format(save_file_dir))
            L.info('Exiting from training early')

    # ------------------------------------------------------------------------------------------ #
    # ----------------------------------- resume train ----------------------------------------- #
    # ------------------------------------------------------------------------------------------ #
    elif args.mode == 1:
        # TODO: catch up with the other parts
        model.load_state_dict(torch.load(args.save))
        load_epoch = int(args.save.split('.')[-1])
        L.info("model restored from epoch-{}: {}".format(load_epoch, args.save))

        try:
            L.info("number of training examples: %d" % t_dataset.len)
            L.info("Reading valid data ...")
            v_dataset = Table2text_seq('valid', type=args.type, USE_CUDA=args.cuda,
                                       batch_size=config.valid_batch, dec_type=args.dec_type)

            L.info("start training...")
            train(t_dataset, v_dataset, model, config.epochs, teacher_forcing_ratio=1, load_epoch=load_epoch)
            writer.close()
        except KeyboardInterrupt:
            L.info('-' * 89)
            L.info('Exiting from training early')
            writer.close()

        dataset = Table2text_seq(args.dataset, type=args.type, USE_CUDA=args.cuda,
                                 batch_size=config.valid_batch, dec_type=args.dec_type)
        L.info("Read $-{}-$ data".format(args.dataset))
        predictor = Predictor(model, dataset.vocab, args.cuda,
                              decoder_type=args.dec_type, unk_gen=config.unk_gen, dataset_type=args.type)
        L.info("number of test examples: %d" % dataset.len)

        L.info("Start Evaluating ...")
        cand, ref, perplexity, others = predictor.preeval_batch(dataset)
        cands_with_unks, cands_with_pgens, cands_ids, tgts_ids, srcs, feats = others

        L.info('Result:')
        L.info('perplexity: {}'.format(perplexity))
        if args.verbose:
            print('\nref[1]: {}'.format(ref[1][0]))
            print('\ncand[1]: {}'.format(cand[1]))
            print('\nsource[1]: {}'.format(srcs[1]))
            for k, v in feats.items():
                print('\n{}[1]: {}'.format(k, v[1]))
            if config.unk_gen:
                print('\ncands_with_unks[1]: {}'.format(cands_with_unks[1]))
            if cands_with_pgens is not None:
                print('\ncands_with_pgens[1]: {}'.format(cands_with_pgens[1]))
            if cands_ids is not None:
                print('\ncands_ids[1]: {}'.format(cands_ids[1]))

        final_scores = metrics.compute_metrics(live=True, cand=cand, ref=ref, epoch=load_epoch,
                                               cands_ids=cands_ids, dataset=dataset, tgts_ids=tgts_ids)

    # --------------------------------------------------------------------------------------- #
    # ----------------------------------- evaluation ---------------------------------------- #
    # --------------------------------------------------------------------------------------- #
    elif args.mode == 2:
        model.load_state_dict(torch.load(args.save))
        load_epoch = int(args.save.split('.')[-1])
        L.info("model restored from epoch-{}: {}".format(load_epoch, args.save))

        dataset = Table2text_seq(args.dataset, type=args.type, USE_CUDA=args.cuda,
                                 batch_size=config.valid_batch, dec_type=args.dec_type)
        print(dataset.sort_indices)
        L.info("Read $-{}-$ data".format(args.dataset))
        predictor = Predictor(model, dataset.vocab, args.cuda,
                              decoder_type=args.dec_type, unk_gen=config.unk_gen, dataset_type=args.type)
        L.info("number of test examples: %d" % dataset.len)

        L.info("Start Evaluating ...")
        cand, ref, perplexity, others = predictor.preeval_batch(dataset, fig=args.fig, save_dir=save_file_dir)
        cands_with_unks, cands_with_pgens, cands_ids, tgts_ids, srcs, feats = others

        L.info('Result:')
        L.info('perplexity: {}'.format(perplexity))
        print('\nsource[1]: {}'.format(srcs[1]))
        if args.verbose:
            for k, v in feats.items():
                print('\n{}[1]: {}'.format(k, v[1]))
            if config.unk_gen:
                print('\ncands_with_unks[1]: {}'.format(cands_with_unks[1]))
            if cands_with_pgens is not None:
                print('\ncands_with_pgens[1]: {}'.format(cands_with_pgens[1]))
            if cands_ids is not None:
                print('\ncands_ids[1]: {}'.format(cands_ids[1]))
            if tgts_ids is not None:
                print('\ntgts_ids[1]: {}'.format(tgts_ids[1]))
            print('\nref[1]: {}'.format(ref[1][0]))
            print('\ncand[1]: {}'.format(cand[1]))

        cand_file_out = "{}/evaluations/{}.epoch_{}.cand.txt".format(save_file_dir, args.dataset, load_epoch)
        with open(cand_file_out, 'w+') as fout:
            for c in range(len(cand)):
                fout.write("{}\n".format(cand[c+1]))

        if cands_with_pgens is not None:
            cand_pgen_file_out = "{}/evaluations/{}.epoch_{}.cand.pgen.txt".format(save_file_dir, args.dataset, load_epoch)
            with open(cand_pgen_file_out, 'w+') as fout:
                for c in range(len(cands_with_pgens)):
                    fout.write("{}\n".format(cands_with_pgens[c+1]))

        if config.unk_gen:
            cand_unk_file_out = "{}/evaluations/{}.epoch_{}.cand.unk.txt".format(save_file_dir, args.dataset, load_epoch)
            with open(cand_unk_file_out, 'w+') as fout:
                for c in range(len(cands_with_unks)):
                    fout.write("{}\n".format(cands_with_unks[c+1]))

        if cands_ids is not None:
            eval_file_out_ids = "{}/evaluations/{}.epoch_{}.cand.ids.txt".format(save_file_dir, args.dataset, load_epoch)
            cands_ids_original = [cands_ids[i+1] for i in dataset.sort_indices]
            with open(eval_file_out_ids, 'w+') as fout:
                for c in range(len(cands_ids_original)):
                    fout.write("{}\n".format(" ".join([str(x) for x in cands_ids_original[c]])))

        ref_file_out = "{}/evaluations/{}.ref.sum.txt".format(save_file_dir, args.dataset)
        with open(ref_file_out, 'w+') as fout:
            for r in range(len(ref)):
                fout.write("{}\n".format(ref[r+1][0]))

        src_file_out = "{}/evaluations/{}.ref.tbv.txt".format(save_file_dir, args.dataset)
        with open(src_file_out, 'w+') as fout:
            for s in range(len(srcs)):
                fout.write("{}\n".format(srcs[s+1]))

        src_file_onmt = "{}/evaluations/{}.ref.tb.onmt.txt".format(save_file_dir, args.dataset)
        with open(src_file_onmt, 'w+') as fout:
            for s in range(len(srcs)):
                a = srcs[s+1].split()
                b = feats['fields'][s+1].split()
                c = feats['rcds'][s+1].split()
                d = feats['has'][s+1].split()
                records = [DELIM.join([h,i,j,k]) for h,i,j,k in zip(a,b,c,d)]
                fout.write("{}\n".format(" ".join(records)))

        fd_file_out = "{}/evaluations/{}.ref.tbl.txt".format(save_file_dir, args.dataset)
        with open(fd_file_out, 'w+') as fout:
            for f in range(len(feats['fields'])):
                fout.write("{}\n".format(feats['fields'][f+1]))

        final_scores = metrics.compute_metrics(live=True, cand=cand, ref=ref, epoch=load_epoch,
                                               cands_ids=cands_ids_original, dataset=dataset, tgts_ids=tgts_ids)
