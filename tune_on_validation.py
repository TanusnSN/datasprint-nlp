import argparse, glob, os, random, joblib, sklearn
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None)
parser.add_argument("--out", default="output")
args, _ = parser.parse_known_args()

MODEL, MAXLEN, BS, EPOCHS, LR, SEED = "intfloat/multilingual-e5-base", 48, 32, 8, 5e-5, 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = "cuda"
os.makedirs(f"{args.out}/model", exist_ok=True)

folder = args.data or os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
train = pd.read_csv(f"{folder}/train.csv")
test = pd.read_csv(f"{folder}/test.csv")
tr, va = train_test_split(train, test_size=0.2, stratify=train["target"], random_state=SEED)

tok = AutoTokenizer.from_pretrained(MODEL)

def encode(texts):
    enc = tok(["query: " + t for t in texts], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(self.enc.config.hidden_size, 56)
    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return self.head(self.drop((h * m).sum(1) / m.sum(1)))

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
svc = LinearSVC().fit(vec.fit_transform(tr["query_text"]), tr["target"])
dv = svc.decision_function(vec.transform(va["query_text"]))

vec_full = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
svc_full = LinearSVC().fit(vec_full.fit_transform(train["query_text"]), train["target"])
dt = svc_full.decision_function(vec_full.transform(test["query_text"]))

net = Net().to(dev)
n_params = sum(p.numel() for p in net.parameters())
svc_params = svc_full.coef_.size + svc_full.intercept_.size
print("e5-base:", n_params, "| tfidf-svc:", svc_params, "| TOTAL:", n_params + svc_params)
assert n_params + svc_params <= 300_000_000
print("sklearn version:", sklearn.__version__)

def batches(ids, mask, y=None, shuffle=False):
    idx = np.random.permutation(len(ids)) if shuffle else np.arange(len(ids))
    for i in range(0, len(idx), BS):
        b = torch.from_numpy(idx[i:i + BS])
        yield ids[b].to(dev), mask[b].to(dev), (torch.tensor(y[b.numpy()]).to(dev) if y is not None else None)

def predict_logits(ids, mask):
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for a, b, _ in batches(ids, mask):
            out.append(net(a, b).float().cpu())
    return torch.cat(out).numpy()

tr_ids, tr_mask = encode(tr["query_text"].tolist())
tr_y = tr["target"].to_numpy()
va_ids, va_mask = encode(va["query_text"].tolist())
te_ids, te_mask = encode(test["query_text"].tolist())

steps = EPOCHS * ((len(tr_ids) + BS - 1) // BS)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler("cuda")
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

for epoch in range(EPOCHS):
    net.train()
    for a, b, y in batches(tr_ids, tr_mask, tr_y, shuffle=True):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(net(a, b).float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
    vl = predict_logits(va_ids, va_mask)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Validation Macro-F1: {f1_score(va['target'], vl.argmax(1), average='macro'):.4f}")

tl = predict_logits(te_ids, te_mask)
e5_val = softmax(vl, axis=1)
best = (0, 5, 0.5)
for T in [2, 3, 5, 8, 12]:
    sv = softmax(dv * T, axis=1)
    for w in np.arange(0.3, 0.91, 0.1):
        f1 = f1_score(va["target"], (w * e5_val + (1 - w) * sv).argmax(1), average="macro")
        if f1 > best[0]:
            best = (f1, T, w)
print("Tuned blend Macro-F1 %.4f at T=%d, weight %.1f" % best)

f1, T, w = best
test["target"] = (w * softmax(tl, axis=1) + (1 - w) * softmax(dt * T, axis=1)).argmax(1)
test[["example_id", "target"]].to_csv(f"{args.out}/submission_final.csv", index=False)

torch.save({k: v.half() for k, v in net.state_dict().items()}, f"{args.out}/model/e5base_fp16.pt")
joblib.dump({"vec": vec_full, "svc": svc_full, "T": T, "w": float(w)}, f"{args.out}/model/tfidf_svc.joblib")
print("Saved submission_final.csv and model/ in", args.out)
