# -*- coding: utf-8 -*-
# probe_mlp_nonlinear.py — нелинейный MLP-зонд per-digit с sham-контролем
import os, torch, torch.nn as nn, numpy as np, random
from sklearn.model_selection import KFold
from jspace import load, DEVICE, DTYPE
from probe import prompt_of, result_of
from probe5 import BIG

class NonLinearDigitsProbe(nn.Module):
    def __init__(self, hidden_dim, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

def collect_digit_samples(model, tok):
    """Для каждого цифрового токена в BIG-трейсах сохраняем (h, digit_label)."""
    Xs, Ys = [], []
    DIGITS = set('0123456789')
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}],
                                      add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=160, do_sample=False)
            hs = model.model(out, output_hidden_states=True).hidden_states[-1][0]
        pl = ids.shape[1]
        for i in range(out.shape[1] - pl):
            txt = tok.decode([out[0, pl + i]]).strip()
            if len(txt) == 1 and txt in DIGITS:           # ← строгая ASCII-проверка
                Xs.append(hs[pl + i].cpu().float())
                Ys.append(int(txt))
    print(f"Собрано {len(Xs)} цифровых токенов из {len(BIG)} промптов")
    return torch.stack(Xs), torch.tensor(Ys, dtype=torch.long)

def train_eval(X, Y, k=4, epochs=300, seed=42):
    """k-fold CV accuracy MLP-зонда."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_accs = []
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr].to(DEVICE), X[te].to(DEVICE)
        Ytr, Yte = Y[tr].to(DEVICE), Y[te].to(DEVICE)
        probe = NonLinearDigitsProbe(X.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        probe.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(probe(Xtr), Ytr); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            preds = probe(Xte).argmax(-1)
            fold_accs.append((preds == Yte).float().mean().item())
    return float(np.mean(fold_accs))

def train_eval_sham(X, Y, k=4, epochs=300, seed=42):
    """Sham: те же данные, перемешанные лейблы."""
    torch.manual_seed(seed + 1); np.random.seed(seed + 1); random.seed(seed + 1)
    perm = torch.randperm(len(Y))
    Y_sham = Y[perm]
    return train_eval(X, Y_sham, k=k, epochs=epochs, seed=seed+100)

def main():
    model, tok = load()
    X, Y = collect_digit_samples(model, tok)
    print(f"Форма: X={tuple(X.shape)}, Y={tuple(Y.shape)}, классов={len(Y.unique())}")
    print(f"Распределение цифр: {torch.bincount(Y).tolist()}")

    print("\nОбучение REAL MLP-зонда (4-fold, 300 эпох × 4 фолда ≈ 2 мин)...")
    acc_real = train_eval(X, Y)
    print(f"  REAL OOS accuracy: {acc_real:.3f}")

    print("\nОбучение SHAM MLP-зонда (перемешанные лейблы)...")
    acc_sham = train_eval_sham(X, Y)
    print(f"  SHAM OOS accuracy: {acc_sham:.3f}")

    base = 1.0 / len(Y.unique())  # случайный бейзлайн ~0.1
    delta = acc_real - acc_sham
    print(f"\nСлучайный бейзлайн: {base:.3f}")
    print(f"REAL:  {acc_real:.3f}")
    print(f"SHAM:  {acc_sham:.3f}")
    print(f"Δ (real − sham): {delta:+.3f}")

    if delta > 0.05:
        print("\n✅ MLP читает per-digit специфично (нелинейная структура есть)")
    elif delta > 0.02:
        print("\n⚠️ Слабый нелинейный сигнал")
    else:
        print("\n❌ MLP не читает per-digit даже нелинейно (седьмой негатив)")

if __name__ == "__main__":
    main()