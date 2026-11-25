import time, joblib, torch, torch.nn as nn, numpy as np, gradio as gr
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModel

MODEL = "intfloat/multilingual-e5-base"
tok = AutoTokenizer.from_pretrained(MODEL)

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

net = Net()
net.load_state_dict(torch.load("model/e5base_fp16.pt", map_location="cpu"))
net.eval()
art = joblib.load("model/tfidf_svc.joblib")
total = sum(p.numel() for p in net.parameters()) + art["svc"].coef_.size + art["svc"].intercept_.size
print("Total inference parameters:", total)
assert total <= 300_000_000

def predict(text):
    start = time.perf_counter()
    enc = tok(["query: " + text], truncation=True, max_length=48, padding=True, return_tensors="pt")
    with torch.no_grad():
        p1 = softmax(net(enc["input_ids"], enc["attention_mask"]).numpy(), axis=1)
    p2 = softmax(art["svc"].decision_function(art["vec"].transform([text])) * art["T"], axis=1)
    p = art["w"] * p1 + (1 - art["w"]) * p2
    ms = (time.perf_counter() - start) * 1000
    top = np.argsort(-p[0])[:3]
    labels = {("Intent %d%s" % (i, " (Out-of-Scope)" if i == 55 else "")): float(p[0][i]) for i in top}
    return labels, "Latency: %.1f ms | Parameters: %d" % (ms, total)

if __name__ == "__main__":
    gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="Voice command (Hindi / Hinglish / English)"),
        outputs=[gr.Label(num_top_classes=3), gr.Textbox(label="Info")],
        title="Data Sprint 2.0: Hindi & Hinglish Intent Classifier",
    ).launch(share=True)
