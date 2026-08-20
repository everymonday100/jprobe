# -*- coding: utf-8 -*-
# probe_mlp_nexttoken.py — честный MLP-зонд: предсказание СЛЕДУЮЩЕЙ цифры
import torch, torch.nn as nn, numpy as np, random
from sklearn.model_selection import KFold
from jspace import load, DEVICE
from probe import prompt_of
from probe5 import BIG

class MLPProbe(nn.Module):
    def __init__(self, hidden_dim, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def collect_nextdigit(model, tok):
    """Собираем пары: hidden state на позиции i → СЛЕДУЮЩАЯ цифра (токен i+1)."""
    DIGITS = set('0123456789')
    Xs, Ys = [], []
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}],
                                      add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=160, do_sample=False)
            hs = model.model(out, output_hidden_states=True).hidden_states[-1][0]
        pl = ids.shape[1]
        gen = out[0, pl:]  # только сгенерированные токены
        # для каждой позиции i предсказываем токен i+1, если i+1 — цифра
        for i in range(len(gen) - 1):
            next_txt = tok.decode([gen[i+1]]).strip()
            if len(next_txt) == 1 and next_txt in DIGITS:
                Xs.append(hs[pl + i].cpu().float())   # hidden state ДО цифры
                Ys.append(int(next_txt))
    print(f"Собрано {len(Xs)} пар (hidden → next digit)")
    return torch.stack(Xs), torch.tensor(Ys, dtype=torch.long)

def train_eval(X, Y, k=4, epochs=300, seed=42, shuffle_labels=False):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if shuffle_labels:
        Y = Y[torch.randperm(len(Y))]
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_accs = []
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr].to(DEVICE), X[te].to(DEVICE)
        Ytr, Yte = Y[tr].to(DEVICE), Y[te].to(DEVICE)
        probe = MLPProbe(X.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        probe.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(probe(Xtr), Ytr); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            fold_accs.append((probe(Xte).argmax(-1) == Yte).float().mean().item())
    return float(np.mean(fold_accs))

def main():
    model, tok = load()
    X, Y = collect_nextdigit(model, tok)
    print(f"X={tuple(X.shape)}, классов={len(Y.unique())}, распр={torch.bincount(Y).tolist()}\n")

    acc_real = train_eval(X, Y)
    print(f"REAL (next-digit) OOS accuracy: {acc_real:.3f}")
    acc_sham = train_eval(X, Y, shuffle_labels=True)
    print(f"SHAM OOS accuracy:               {acc_sham:.3f}")

    base = 1.0 / len(Y.unique())
    delta = acc_real - acc_sham
    print(f"\nБейзлайн: {base:.3f}")
    print(f"Δ (real − sham): {delta:+.3f}")

    if delta > 0.3:
        print("\n✅ Настоящий predictive сигнал: модель заранее знает следующую цифру")
    elif delta > 0.1:
        print("\n⚠️ Умеренный predictive сигнал")
    else:
        print("\n❌ Предыдущий результат 1.000 был утечкой через residual connection")

if __name__ == "__main__":
    main()