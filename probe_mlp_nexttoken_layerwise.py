# -*- coding: utf-8 -*-
# probe_mlp_nexttoken_layerwise.py — layer-wise next-digit prediction
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

def collect_nextdigit_layerwise(model, tok, layers):
    """Собираем (hidden state слой L на позиции i) → СЛЕДУЮЩАЯ цифра (токен i+1)."""
    DIGITS = set('0123456789')
    data = {l: {'X': [], 'Y': []} for l in layers}
    
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}],
                                      add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=160, do_sample=False)
            hs_all = model.model(out, output_hidden_states=True).hidden_states
        
        pl = ids.shape[1]
        gen = out[0, pl:]
        for i in range(len(gen) - 1):
            next_txt = tok.decode([gen[i+1]]).strip()
            if len(next_txt) == 1 and next_txt in DIGITS:
                digit = int(next_txt)
                for l in layers:
                    data[l]['X'].append(hs_all[l][0, pl + i].cpu().float())
                    data[l]['Y'].append(digit)
    
    for l in layers:
        data[l]['X'] = torch.stack(data[l]['X'])
        data[l]['Y'] = torch.tensor(data[l]['Y'], dtype=torch.long)
    return data

def train_eval(X, Y, k=4, epochs=200, seed=42, shuffle=False):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if shuffle: Y = Y[torch.randperm(len(Y))]
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    accs = []
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr].to(DEVICE), X[te].to(DEVICE)
        Ytr, Yte = Y[tr].to(DEVICE), Y[te].to(DEVICE)
        probe = MLPProbe(X.shape[1]).to(DEVICE)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss()
        probe.train()
        for _ in range(epochs):
            opt.zero_grad(); loss = loss_fn(probe(Xtr), Ytr); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad(): accs.append((probe(Xte).argmax(-1) == Yte).float().mean().item())
    return float(np.mean(accs))

def main():
    model, tok = load()
    layers = list(range(0, 36, 4)) + [35]
    
    print(f"Сбор next-digit пар со всех слоёв (64 промпта)...")
    data = collect_nextdigit_layerwise(model, tok, layers)
    n = len(data[layers[0]]['Y'])
    print(f"Собрано {n} пар\n")
    
    print(f"{'Layer':>5s} | {'REAL':>6s} {'SHAM':>6s} {'Δ':>7s} | verdict")
    print("-"*50)
    
    results = []
    for L in layers:
        X, Y = data[L]['X'], data[L]['Y']
        acc_real = train_eval(X, Y, epochs=150)
        acc_sham = train_eval(X, Y, epochs=150, shuffle=True)
        delta = acc_real - acc_sham
        results.append((L, acc_real, acc_sham, delta))
        marker = "✅" if delta > 0.3 else ("⚠️" if delta > 0.1 else "❌")
        print(f"{L:>5d} | {acc_real:>6.3f} {acc_sham:>6.3f} {delta:>+7.3f} | {marker}")
    
    print("\n" + "="*50)
    print("Динамика next-digit prediction:")
    
    early = [r for r in results if r[0] <= 12]
    mid = [r for r in results if 13 <= r[0] <= 27]
    late = [r for r in results if r[0] >= 28]
    
    early_delta = np.mean([r[3] for r in early]) if early else 0
    mid_delta = np.mean([r[3] for r in mid]) if mid else 0
    late_delta = np.mean([r[3] for r in late]) if late else 0
    
    print(f"  Ранние (L0-12):   Δ={early_delta:+.3f}")
    print(f"  Средние (L13-27): Δ={mid_delta:+.3f}")
    print(f"  Поздние (L28-35): Δ={late_delta:+.3f}")
    
    if early_delta > 0.5:
        print("\n🎯 Цифры планируются в ранних слоях — W-форма опровергнута!")
    elif mid_delta > 0.5 and early_delta < 0.3:
        print("\n✅ Цифры возникают в фазе кипения — согласуется с W-формой")
    elif late_delta > 0.5 and mid_delta < 0.3:
        print("\n✅ Цифры кристаллизуются перед unembedding — согласуется с GWT")
    else:
        print("\n⚠️ Predictive сигнал распределён по всем слоям")

if __name__ == "__main__":
    main()