import re
import sys
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────
CFG = dict(
    # paths
    train_path   = "train.csv",
    test_path    = "test.csv",
    output_path  = "submission_lstm.csv",

    # character CNN
    char_embed_dim  = 32,
    char_filters    = [64, 64],
    char_kernels    = [3, 5],
    max_word_chars  = 30,

    # numeric word features (non-char)
    n_word_features = 20,

    # BiLSTM
    lstm_hidden    = 128,
    lstm_layers    = 2,
    lstm_dropout   = 0.25,

    # head
    head_hidden    = 64,
    dropout        = 0.30,

    # training
    batch_size     = 32,          # sequences per batch
    epochs         = 40,
    lr             = 3e-3,
    weight_decay   = 1e-4,
    patience       = 8,           # early stopping

    # loss: MAE is more robust to outliers than MSE for TRT
    # We also add a BCE term for the skip gate
    lambda_gate    = 0.3,         # weight of skip-gate BCE loss

    # cross-validation
    n_folds        = 5,

    # skip threshold (will be tuned on OOF)
    skip_thresh    = 0.50,

    seed           = 42,
    device         = "cuda" if torch.cuda.is_available() else "cpu",
)

torch.manual_seed(CFG['seed'])
np.random.seed(CFG['seed'])
print(f"Device: {CFG['device']}")


# ──────────────────────────────────────────────────────────────────────────
# METRIC
# ──────────────────────────────────────────────────────────────────────────
def eval_metric(y_true, preds):
    y_true = np.array(y_true, dtype=float)
    preds  = np.array(preds,  dtype=float)
    r2     = max(0.0, r2_score(y_true, preds))
    pr     = pearsonr(y_true, preds)[0]
    if np.isnan(pr): pr = 0.0
    return 100.0 * (abs(pr) + r2) / 2.0


# ──────────────────────────────────────────────────────────────────────────
# 1.  LOAD & PARSE
# ──────────────────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(CFG['train_path'])
test_df  = pd.read_csv(CFG['test_path'])

def parse_wid(wid):
    parts    = wid.split('_')
    word_pos = int(parts[-1])
    page_num = int(parts[-2]) if parts[-2].isdigit() else 1
    return word_pos, page_num

for df in [train_df, test_df]:
    parsed          = df['word_id'].apply(parse_wid)
    df['word_pos']  = [x[0] for x in parsed]
    df['page_num']  = [x[1] for x in parsed]
    df['word_norm'] = df['word'].apply(lambda w: re.sub(r'[^\w]', '', str(w).lower()))

print(f"Train: {train_df.shape}   Test: {test_df.shape}")


# ──────────────────────────────────────────────────────────────────────────
# 2.  CORPUS STATISTICS  (computed on train, applied to both)
# ──────────────────────────────────────────────────────────────────────────
nz          = train_df[train_df['answer'] > 0]
word_mean   = nz.groupby('word_norm')['answer'].mean()
word_median = nz.groupby('word_norm')['answer'].median()
word_count  = nz.groupby('word_norm')['answer'].count()
word_freq   = train_df['word_norm'].value_counts()

total_obs   = train_df.groupby('word_norm')['answer'].count()
train_df['_z'] = (train_df['answer'] == 0).astype(int)
zero_obs    = train_df.groupby('word_norm')['_z'].sum()
word_skip   = zero_obs / total_obs
train_df.drop(columns=['_z'], inplace=True)

GLOBAL_MEAN   = float(nz['answer'].mean())       # ≈381 ms
GLOBAL_MEDIAN = float(nz['answer'].median())     # ≈286 ms
GLOBAL_SKIP   = float((train_df['answer']==0).mean())  # ≈0.306
TOTAL_TOKENS  = len(train_df)

# Participant speed (train only) – will be used as conditioning scalar
part_speed_raw = nz.groupby('participant_id')['answer'].mean()
# Normalize: speed_factor = participant_mean / global_mean
# fast readers → <1.0, slow readers → >1.0
part_speed_factor = part_speed_raw / GLOBAL_MEAN
print(f"Participant speed factors: min={part_speed_factor.min():.2f}  "
      f"max={part_speed_factor.max():.2f}")


# ──────────────────────────────────────────────────────────────────────────
# 3.  CHARACTER ALPHABET
# ──────────────────────────────────────────────────────────────────────────
# Build from all chars seen in train + test
all_text = pd.concat([train_df['word'], test_df['word']]).astype(str)
CHARS    = ['<PAD>', '<UNK>'] + sorted(set(''.join(all_text.tolist())))
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
print(f"Character vocabulary size: {VOCAB_SIZE}")


def word_to_char_ids(word: str, max_len: int) -> list:
    ids = [CHAR2IDX.get(c, CHAR2IDX['<UNK>']) for c in str(word)[:max_len]]
    ids += [CHAR2IDX['<PAD>']] * (max_len - len(ids))
    return ids


# ──────────────────────────────────────────────────────────────────────────
# 4.  NUMERIC WORD FEATURES  (same as LightGBM model – proven correlations)
# ──────────────────────────────────────────────────────────────────────────
RO_FUNCTION_WORDS = set((
    "de si in a cu sa care pe la se din ca au mai nu este un o sunt fi "
    "el ea ei ele al ai ale lui lor ne le ni li vi am voi noi "
    "dar sau ori ci insa totusi deci astfel i e ce cine cand unde cum cat "
    "cel cea cei cele il ii m l n v ma te va s "
    "pentru prin spre fata langa sub fara intre dupa vor va "
).split())

def extract_word_features(df: pd.DataFrame) -> np.ndarray:
    """Returns array of shape (N, n_word_features) – one row per word."""
    wn  = df['word_norm']
    w   = df['word'].astype(str)

    wmv   = wn.map(word_mean).fillna(GLOBAL_MEAN).values
    wsv   = wn.map(word_skip).fillna(GLOBAL_SKIP).values
    wcv   = wn.map(word_count).fillna(0).values
    wfv   = wn.map(word_freq).fillna(0.5).values
    alpha = np.clip(wcv, 0, 30) / 30.0
    wblend= alpha * wmv + (1 - alpha) * GLOBAL_MEAN

    wlen  = w.str.len().values
    wclen = wn.str.len().values
    nsyl  = np.array([max(1, len(re.findall(r'[aeiouăâî]+', x))) for x in wn])
    ndiac = np.array([sum(1 for c in x if c in 'ăâîșțĂÂÎȘȚ') for x in w])
    is_fw = np.array([float(x in RO_FUNCTION_WORDS) for x in wn])
    is_pu = (wn.str.len() == 0).values.astype(float)
    is_ca = np.array([float(len(x)>0 and x[0].isupper()) for x in w])
    logf  = np.log1p(wfv)
    logwp = np.log1p(df['word_pos'].values)

    # Normalize continuous features to ~[0,1] range
    feat = np.column_stack([
        wlen   / 20.0,              # 0  word length
        wclen  / 20.0,              # 1  clean length
        nsyl   / 8.0,               # 2  syllables
        ndiac  / 5.0,               # 3  diacritics
        is_fw,                      # 4  function word flag
        is_pu,                      # 5  punctuation flag
        is_ca,                      # 6  capitalized
        logf   / 10.0,              # 7  log frequency
        logwp  / 5.0,               # 8  log word position
        wsv,                        # 9  skip rate (already in [0,1])
        wmv    / 1000.0,            # 10 word mean TRT (norm)
        wcv    / 30.0,              # 11 observation count
        (wcv > 0).astype(float),    # 12 has word stats
        wblend / 1000.0,            # 13 blended mean TRT
        # suffix features (Romanian morphology)
        np.array([float(x.endswith('ului')) for x in wn]),  # 14
        np.array([float(x.endswith('ilor')) for x in wn]),  # 15
        np.array([float(x.endswith('are'))  for x in wn]),  # 16
        np.array([float(x.endswith('ere'))  for x in wn]),  # 17
        np.array([float(x.endswith('ând') or x.endswith('ind')) for x in wn]),  # 18
        (df['page_num'].values / 20.0),                      # 19
    ]).astype(np.float32)

    assert feat.shape[1] == CFG['n_word_features'], \
        f"Feature mismatch: {feat.shape[1]} vs {CFG['n_word_features']}"
    return feat


# ──────────────────────────────────────────────────────────────────────────
# 5.  DATASET  –  one sample = one reading sequence (one page, one participant)
# ──────────────────────────────────────────────────────────────────────────
class ReadingSequenceDataset(Dataset):
    """
    Each item is one reading session: a participant reading one page.
    Returns:
        char_ids   – (seq_len, max_word_chars)  long
        word_feats – (seq_len, n_word_features)  float
        speed      – scalar float  (participant speed factor, 1.0 for unknown)
        targets    – (seq_len,) float  [TRT values, 0 = skipped]
        mask       – (seq_len,) bool  [True = valid position]
    """
    def __init__(self, df: pd.DataFrame, is_train: bool = True):
        self.sequences = []
        self.is_train  = is_train

        # Precompute word features for the whole dataframe
        word_feats_all = extract_word_features(df)
        df = df.copy()
        df['_feat_idx'] = np.arange(len(df))

        # Group into sequences: (text, participant_id, page_num)
        grouped = df.groupby(['text', 'participant_id', 'page_num'])
        for (text, pid, page), grp in grouped:
            grp = grp.sort_values('word_pos').reset_index(drop=True)
            seq_len = len(grp)

            # Character IDs for each word
            char_ids = np.array([
                word_to_char_ids(w, CFG['max_word_chars'])
                for w in grp['word']
            ], dtype=np.int64)   # (seq_len, max_word_chars)

            # Numeric features
            feat_idx  = grp['_feat_idx'].values
            word_feats = word_feats_all[feat_idx]  # (seq_len, n_feats)

            # Participant speed factor (1.0 for unseen test participants)
            speed = float(part_speed_factor.get(pid, 1.0))

            # Targets
            if is_train:
                targets = grp['answer'].values.astype(np.float32)
            else:
                targets = np.zeros(seq_len, dtype=np.float32)  # placeholder

            self.sequences.append({
                'char_ids':   char_ids,
                'word_feats': word_feats,
                'speed':      speed,
                'targets':    targets,
                'pid':        pid,
                'text':       text,
                'page':       page,
                'word_pos':   grp['word_pos'].values,
                'datapointID': grp.get('datapointID', pd.Series([None]*seq_len)).values
                                if 'datapointID' in grp.columns else np.array([None]*seq_len),
                'seq_len':    seq_len,
            })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_fn(batch):
    """Pad sequences to the longest in the batch."""
    seq_lens = [b['seq_len'] for b in batch]
    max_len  = max(seq_lens)

    char_ids_list  = []
    word_feats_list= []
    targets_list   = []
    speed_list     = []
    mask_list      = []

    for b in batch:
        L = b['seq_len']
        pad = max_len - L

        ci = torch.tensor(b['char_ids'], dtype=torch.long)  # (L, C)
        wf = torch.tensor(b['word_feats'], dtype=torch.float32)  # (L, F)
        tg = torch.tensor(b['targets'], dtype=torch.float32)     # (L,)
        mk = torch.ones(L, dtype=torch.bool)

        if pad > 0:
            ci = F.pad(ci, (0, 0, 0, pad))      # pad seq dim
            wf = F.pad(wf, (0, 0, 0, pad))
            tg = F.pad(tg, (0, pad))
            mk = F.pad(mk, (0, pad), value=False)

        char_ids_list.append(ci)
        word_feats_list.append(wf)
        targets_list.append(tg)
        speed_list.append(b['speed'])
        mask_list.append(mk)

    return {
        'char_ids':   torch.stack(char_ids_list),   # (B, T, C)
        'word_feats': torch.stack(word_feats_list), # (B, T, F)
        'speeds':     torch.tensor(speed_list, dtype=torch.float32),  # (B,)
        'targets':    torch.stack(targets_list),    # (B, T)
        'mask':       torch.stack(mask_list),        # (B, T)
        'seq_lens':   torch.tensor(seq_lens, dtype=torch.long),
    }


# ──────────────────────────────────────────────────────────────────────────
# 6.  MODEL
# ──────────────────────────────────────────────────────────────────────────
class CharCNN(nn.Module):
    """
    Encodes a word from its characters.
    Input:  (B, T, C) – batch, time(words), chars
    Output: (B, T, char_out_dim)
    """
    def __init__(self, vocab_size, embed_dim, filters, kernels, max_chars):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs  = nn.ModuleList([
            nn.Conv1d(embed_dim, f, k, padding=k//2)
            for f, k in zip(filters, kernels)
        ])
        self.out_dim = sum(filters)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        x = x.view(B * T, C)                     # (B*T, C)
        e = self.embed(x)                         # (B*T, C, embed)
        e = e.transpose(1, 2)                     # (B*T, embed, C)
        pooled = []
        for conv in self.convs:
            h = F.gelu(conv(e))                   # (B*T, filters, C)
            h = h.max(dim=2).values               # (B*T, filters)  – global max pool
            pooled.append(h)
        out = torch.cat(pooled, dim=1)            # (B*T, sum_filters)
        out = out.view(B, T, -1)                  # (B, T, sum_filters)
        return out


class EyeTrackingLSTM(nn.Module):
    """
    Full model:
      CharCNN → concat word_features → BiLSTM →
      speed-conditioned head → (gate, trt)

    Output:
      gate – (B, T) probability of NOT skipping (sigmoid)
      trt  – (B, T) predicted TRT in ms (softplus to ensure positivity)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Character encoder
        self.char_cnn = CharCNN(
            vocab_size = VOCAB_SIZE,
            embed_dim  = cfg['char_embed_dim'],
            filters    = cfg['char_filters'],
            kernels    = cfg['char_kernels'],
            max_chars  = cfg['max_word_chars'],
        )
        char_dim = self.char_cnn.out_dim  # 128

        # Input to LSTM: char features + numeric word features
        lstm_in_dim = char_dim + cfg['n_word_features']

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size   = lstm_in_dim,
            hidden_size  = cfg['lstm_hidden'],
            num_layers   = cfg['lstm_layers'],
            batch_first  = True,
            bidirectional= True,
            dropout      = cfg['lstm_dropout'] if cfg['lstm_layers'] > 1 else 0.0,
        )
        lstm_out_dim = cfg['lstm_hidden'] * 2  # bidirectional

        # Speed adapter: one linear layer that modulates LSTM output
        # speed_factor ∈ [0.65, 1.55] → embed it into a scaling vector
        self.speed_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, lstm_out_dim),
            nn.Sigmoid(),   # output in (0,1) → acts as gating/scaling
        )

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(cfg['dropout']),
            nn.Linear(lstm_out_dim, cfg['head_hidden']),
            nn.GELU(),
            nn.Dropout(cfg['dropout'] * 0.5),
        )

        # Skip gate head: P(word is fixated | word features, context)
        self.gate_head = nn.Linear(cfg['head_hidden'], 1)

        # TRT regression head: predicted reading time
        self.trt_head = nn.Linear(cfg['head_hidden'], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)

    def forward(self, char_ids, word_feats, speeds, seq_lens):
        """
        char_ids  : (B, T, C)
        word_feats: (B, T, F)
        speeds    : (B,)
        seq_lens  : (B,)
        """
        # Character CNN
        char_enc = self.char_cnn(char_ids)      # (B, T, char_dim)

        # Concat with numeric features
        x = torch.cat([char_enc, word_feats], dim=-1)  # (B, T, lstm_in)

        # BiLSTM (packed for efficiency)
        packed = pack_padded_sequence(
            x, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # (B, T, 2H)

        # Speed modulation: scale hidden state by participant speed vector
        sp = speeds.unsqueeze(1)                          # (B, 1)
        speed_gate = self.speed_embed(sp).unsqueeze(1)    # (B, 1, 2H)
        lstm_out = lstm_out * speed_gate                  # (B, T, 2H)

        # Shared trunk
        h = self.trunk(lstm_out)                          # (B, T, head_hidden)

        # Dual heads
        gate = torch.sigmoid(self.gate_head(h)).squeeze(-1)  # (B, T) ∈ (0,1)
        trt  = F.softplus(self.trt_head(h)).squeeze(-1)     # (B, T) > 0

        # Scale TRT to ms range (softplus output ≈1, we want ~300ms)
        trt = trt * 200.0 + 100.0   # roughly (100, ∞) ms

        return gate, trt


# ──────────────────────────────────────────────────────────────────────────
# 7.  LOSS FUNCTION
# ──────────────────────────────────────────────────────────────────────────
def hurdle_loss(gate, trt_pred, targets, mask, lambda_gate=0.3):
    """
    Hurdle model loss:
      - BCE on gate: predict whether word is fixated (target > 0)
      - MAE on TRT: only for fixated words
      - Final = MAE + lambda_gate * BCE
    """
    gate_target = (targets > 0).float()   # 1=fixated, 0=skipped

    # BCE for skip gate (all positions)
    bce = F.binary_cross_entropy(gate[mask], gate_target[mask])

    # MAE on non-zero targets only
    nz_mask = mask & (targets > 0)
    if nz_mask.sum() == 0:
        return bce

    # Normalize TRT to [0,1]-ish range for stable gradients
    trt_norm_pred  = trt_pred[nz_mask] / 1000.0
    trt_norm_true  = targets[nz_mask]  / 1000.0

    mae  = F.l1_loss(trt_norm_pred, trt_norm_true)

    # Pearson-correlation loss (maximise correlation)
    if nz_mask.sum() > 1:
        yp = trt_norm_pred - trt_norm_pred.mean()
        yt = trt_norm_true - trt_norm_true.mean()
        pearson_loss = 1.0 - (yp * yt).sum() / (
            yp.norm() * yt.norm() + 1e-8
        )
    else:
        pearson_loss = torch.tensor(0.0)

    return mae + 0.3 * pearson_loss + lambda_gate * bce


# ──────────────────────────────────────────────────────────────────────────
# 8.  TRAIN / EVAL FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, lambda_gate):
    model.train()
    total_loss = 0.0
    for batch in loader:
        char_ids   = batch['char_ids'].to(device)
        word_feats = batch['word_feats'].to(device)
        speeds     = batch['speeds'].to(device)
        targets    = batch['targets'].to(device)
        mask       = batch['mask'].to(device)
        seq_lens   = batch['seq_lens']

        optimizer.zero_grad()
        gate, trt = model(char_ids, word_feats, speeds, seq_lens)
        loss = hurdle_loss(gate, trt, targets, mask, lambda_gate)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def predict(model, loader, device, skip_thresh=0.50):
    """Returns flat arrays of (predictions, targets) in original order."""
    model.eval()
    all_preds   = []
    all_targets = []
    all_meta    = []  # (pid, text, page, word_pos) for reconstruction

    for batch in loader:
        char_ids   = batch['char_ids'].to(device)
        word_feats = batch['word_feats'].to(device)
        speeds     = batch['speeds'].to(device)
        seq_lens   = batch['seq_lens']
        targets    = batch['targets']
        mask       = batch['mask']

        gate, trt = model(char_ids, word_feats, speeds, seq_lens)
        gate = gate.cpu().numpy()
        trt  = trt.cpu().numpy()
        mask = mask.numpy()

        for i, b in enumerate(loader.dataset.sequences[
                len(all_meta): len(all_meta) + len(batch['seq_lens'])
            ]):
            pass  # we'll reconstruct below

        # Flatten valid positions
        for i in range(len(seq_lens)):
            L    = seq_lens[i].item()
            g    = gate[i, :L]
            t    = trt[i, :L]
            targ = targets[i, :L].numpy()
            # Hurdle: if gate < thresh → predict 0 (skipped)
            pred = np.where(g < skip_thresh, 0.0, t)
            all_preds.append(pred)
            all_targets.append(targ)

    return np.concatenate(all_preds), np.concatenate(all_targets)


# ──────────────────────────────────────────────────────────────────────────
# 9.  SPEED ESTIMATION FOR UNSEEN TEST PARTICIPANTS
#     Strategy: run the model on the first ~20% of words of each test
#     sequence with skip_thresh=0 (predict TRT for all), then compute
#     the ratio of predicted/actual implied speed and adjust.
#     
#     Since we don't have test labels, we use a simpler proxy:
#     estimate speed from the distribution of word lengths & types.
#     In practice: default to 1.0 and let the model generalize.
#     A better approach would be to do online adaptation if partial
#     labels are available.
# ──────────────────────────────────────────────────────────────────────────
# For this competition: test participants are unseen (speed=1.0 default)
# The BiLSTM's learned priors about word difficulty still apply.
# The speed_embed with input 1.0 acts as a "average reader" prior.


# ──────────────────────────────────────────────────────────────────────────
# 10. CROSS-VALIDATION  (5-fold by participant)
# ──────────────────────────────────────────────────────────────────────────
participants = np.array(sorted(train_df['participant_id'].unique()))
np.random.seed(CFG['seed'])
np.random.shuffle(participants)
fold_assign  = {p: i % CFG['n_folds'] for i, p in enumerate(participants)}
fold_ids     = train_df['participant_id'].map(fold_assign).values

device = torch.device(CFG['device'])

# We need to store OOF predictions aligned to train_df rows
# Each train row corresponds to one (text, participant, page, word_pos)
# We'll index by the sorted order within each sequence

print("\n=== BUILDING DATASETS ===")
train_dataset = ReadingSequenceDataset(train_df, is_train=True)
print(f"Train sequences: {len(train_dataset)}")

# Map each sequence back to train_df indices for OOF alignment
# Build a lookup: (text, pid, page, word_pos) → train_df row index
train_df['_row_idx'] = np.arange(len(train_df))
row_lookup = {}
for idx, row in train_df.iterrows():
    key = (row['text'], row['participant_id'], row['page_num'], row['word_pos'])
    row_lookup[key] = row['_row_idx']

oof_pred     = np.full(len(train_df), np.nan)
oof_gate_prob= np.full(len(train_df), np.nan)
oof_trt_pred = np.full(len(train_df), np.nan)

# Collect test predictions across folds (average)
test_dataset = ReadingSequenceDataset(test_df, is_train=False)
te_gate_acc  = np.zeros(len(test_df))
te_trt_acc   = np.zeros(len(test_df))

print("\n=== 5-FOLD CROSS VALIDATION ===")
for fi in range(CFG['n_folds']):
    val_parts = participants[fi::CFG['n_folds']]
    print(f"\n--- Fold {fi+1}/{CFG['n_folds']}  "
          f"(val participants: {sorted(val_parts.tolist())}) ---")

    # Split sequences by participant
    tr_seqs  = [s for s in train_dataset.sequences
                if s['pid'] not in set(val_parts.tolist())]
    val_seqs = [s for s in train_dataset.sequences
                if s['pid'] in set(val_parts.tolist())]

    # Temporarily swap dataset sequences for DataLoader
    class SubDataset(Dataset):
        def __init__(self, seqs): self.sequences = seqs
        def __len__(self): return len(self.sequences)
        def __getitem__(self, i): return self.sequences[i]

    tr_loader  = DataLoader(SubDataset(tr_seqs),  batch_size=CFG['batch_size'],
                            shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(SubDataset(val_seqs), batch_size=CFG['batch_size'],
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    te_loader  = DataLoader(test_dataset,         batch_size=CFG['batch_size'],
                            shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Fresh model per fold
    model = EyeTrackingLSTM(CFG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs'], eta_min=CFG['lr'] * 0.05
    )

    best_val_score = -np.inf
    patience_cnt   = 0
    best_state     = None

    for epoch in range(CFG['epochs']):
        train_loss = train_epoch(
            model, tr_loader, optimizer, device, CFG['lambda_gate']
        )
        scheduler.step()

        # Validate
        val_preds, val_targets = predict(model, val_loader, device, CFG['skip_thresh'])
        val_score = eval_metric(val_targets, val_preds)

        print(f"  epoch {epoch+1:3d}/{CFG['epochs']}  "
              f"loss={train_loss:.4f}  val_score={val_score:.2f}", flush=True)

        if val_score > best_val_score:
            best_val_score = val_score
            patience_cnt   = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= CFG['patience']:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best weights and collect OOF + test predictions
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  Best val score: {best_val_score:.2f}")

    # OOF: collect per-word predictions and map back to train_df rows
    model.eval()
    with torch.no_grad():
        for seq in val_seqs:
            ci  = torch.tensor(seq['char_ids'],   dtype=torch.long).unsqueeze(0).to(device)
            wf  = torch.tensor(seq['word_feats'], dtype=torch.float32).unsqueeze(0).to(device)
            sp  = torch.tensor([seq['speed']],    dtype=torch.float32).to(device)
            sl  = torch.tensor([seq['seq_len']],  dtype=torch.long)
            gate, trt = model(ci, wf, sp, sl)
            gate = gate.cpu().numpy()[0]
            trt  = trt.cpu().numpy()[0]
            for j, wp in enumerate(seq['word_pos']):
                key = (seq['text'], seq['pid'], seq['page'], wp)
                if key in row_lookup:
                    ridx = row_lookup[key]
                    oof_gate_prob[ridx] = gate[j]
                    oof_trt_pred[ridx]  = trt[j]
                    oof_pred[ridx]      = 0.0 if gate[j] < CFG['skip_thresh'] else trt[j]

    # Test predictions (accumulate across folds)
    with torch.no_grad():
        te_gate_fold = np.zeros(len(test_df))
        te_trt_fold  = np.zeros(len(test_df))
        te_idx = 0
        for seq in test_dataset.sequences:
            ci  = torch.tensor(seq['char_ids'],   dtype=torch.long).unsqueeze(0).to(device)
            wf  = torch.tensor(seq['word_feats'], dtype=torch.float32).unsqueeze(0).to(device)
            sp  = torch.tensor([seq['speed']],    dtype=torch.float32).to(device)
            sl  = torch.tensor([seq['seq_len']],  dtype=torch.long)
            gate, trt = model(ci, wf, sp, sl)
            g = gate.cpu().numpy()[0]
            t = trt.cpu().numpy()[0]
            L = seq['seq_len']
            te_gate_fold[te_idx:te_idx+L] = g
            te_trt_fold[te_idx:te_idx+L]  = t
            te_idx += L
    te_gate_acc += te_gate_fold / CFG['n_folds']
    te_tr_acc   += te_trt_fold  / CFG['n_folds']


# ──────────────────────────────────────────────────────────────────────────
# 11. OOF EVALUATION & THRESHOLD TUNING
# ──────────────────────────────────────────────────────────────────────────
valid_mask = ~np.isnan(oof_pred)
y_true     = train_df['answer'].values.astype(float)

print(f"\n=== OOF RESULTS (t=0.50) ===")
oof_score = eval_metric(y_true[valid_mask], oof_pred[valid_mask])
print(f"OOF score: {oof_score:.2f}")

print("\n=== THRESHOLD SEARCH ===")
best_t, best_s = 0.50, 0.0
for t in np.arange(0.15, 0.90, 0.05):
    p   = np.where(oof_gate_prob[valid_mask] < t, 0.0, oof_trt_pred[valid_mask])
    s   = eval_metric(y_true[valid_mask], p)
    r2v = max(0, r2_score(y_true[valid_mask], p))
    prv = abs(pearsonr(y_true[valid_mask], p)[0])
    mk  = "  <<<" if s > best_s else ""
    print(f"  t={t:.2f}  score={s:.2f}  R²={r2v:.3f}  Pearson={prv:.3f}{mk}")
    if s > best_s:
        best_s, best_t = s, t

print(f"\nBest threshold: t={best_t:.2f}  →  score={best_s:.2f}")


# ──────────────────────────────────────────────────────────────────────────
# 12. SAVE SUBMISSION
# ──────────────────────────────────────────────────────────────────────────
# Rebuild test predictions with best threshold
# We need to map te_gate_acc / te_tr_acc back to test_df rows
# The test_dataset iterates sequences in a fixed order → te_idx tracking

te_pred_final = np.zeros(len(test_df))
te_idx = 0
for seq in test_dataset.sequences:
    L = seq['seq_len']
    g = te_gate_acc[te_idx:te_idx+L]
    t = te_tr_acc[te_idx:te_idx+L]
    pred = np.where(g < best_t, 0.0, t)
    te_pred_final[te_idx:te_idx+L] = pred
    te_idx += L

# Clip to valid range
te_pred_final = np.where(te_pred_final > 0,
                         np.clip(te_pred_final, 100, 5000),
                         0.0)

sub = test_df[['datapointID']].copy()
sub['subtaskID'] = 1
sub['answer']    = np.round(te_pred_final).astype(int)
sub = sub[['subtaskID', 'datapointID', 'answer']]
sub.to_csv(CFG['output_path'], index=False)

print(f"\n=== SUBMISSION ===")
print(f"Saved: {CFG['output_path']}  ({len(sub)} rows)")
print(f"Zeros: {(sub['answer']==0).sum()}  ({(sub['answer']==0).mean():.1%})")
print(f"Mean (non-zero): {sub[sub['answer']>0]['answer'].mean():.1f} ms")
print(sub.head(15).to_string(index=False))
print(f"\n>>> FINAL OOF SCORE: {best_s:.2f} / 100")
print("Done.")
