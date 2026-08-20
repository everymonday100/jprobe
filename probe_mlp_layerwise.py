# -*- coding: utf-8 -*-
# probe_mlp_layerwise.py — layer-wise MLP digit readout
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

def collect_digit_samples_layerwise(model, tok, layers):
    """Собираем цифровые токены со ВСЕХ слоёв."""
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
        for i in range(out.shape[1] - pl):
            txt = tok.decode([out[0, pl + i]]).strip()
            if len(txt) == 1 and txt in DIGITS:
                digit = int(txt)
                for l in layers:
                    data[l]['X'].append(hs_all[l][0, pl + i].cpu().float())
                    data[l]['Y'].append(digit)
    
    for l in layers:
        data[l]['X'] = torch.stack(data[l]['X'])
        data[l]['Y'] = torch.tensor(data[l]['Y'], dtype=torch.long)
    return data

def train_eval_mlp(X, Y, k=4, epochs=300, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
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
            preds = probe(Xte).argmax(-1)
            fold_accs.append((preds == Yte).float().mean().item())
    return float(np.mean(fold_accs))

def train_eval_sham(X, Y, k=4, epochs=300, seed=42):
    perm = torch.randperm(len(Y))
    Y_sham = Y[perm]
    return train_eval_mlp(X, Y_sham, k=k, epochs=epochs, seed=seed+100)

def main():
    model, tok = load()
    layers = list(range(0, 36, 4)) + [35]  # 0, 4, 8, ..., 32, 35
    
    print(f"Сбор цифровых токенов со всех слоёв (64 промпта)...")
    data = collect_digit_samples_layerwise(model, tok, layers)
    n_samples = len(data[layers[0]]['Y'])
    print(f"Собрано {n_samples} цифровых токенов\n")
    
    print(f"{'Layer':>5s} | {'REAL':>6s} {'SHAM':>6s} {'Δ':>7s} | verdict")
    print("-"*50)
    
    results = []
    for L in layers:
        X, Y = data[L]['X'], data[L]['Y']
        acc_real = train_eval_mlp(X, Y, epochs=200)  # сокращаем для скорости
        acc_sham = train_eval_sham(X, Y, epochs=200)
        delta = acc_real - acc_sham
        results.append((L, acc_real, acc_sham, delta))
        marker = "✅" if delta > 0.1 else ("⚠️" if delta > 0.05 else "❌")
        print(f"{L:>5d} | {acc_real:>6.3f} {acc_sham:>6.3f} {delta:>+7.3f} | {marker}")
    
    print("\n" + "="*50)
    print("Анализ динамики:")
    
    # Ранние слои (0-12)
    early = [r for r in results if r[0] <= 12]
    early_acc = np.mean([r[1] for r in early])
    early_delta = np.mean([r[3] for r in early])
    print(f"  Ранние (L0-12):   REAL={early_acc:.3f}, Δ={early_delta:+.3f}")
    
    # Средние слои (13-27)
    mid = [r for r in results if 13 <= r[0] <= 27]
    mid_acc = np.mean([r[1] for r in mid])
    mid_delta = np.mean([r[3] for r in mid])
    print(f"  Средние (L13-27): REAL={mid_acc:.3f}, Δ={mid_delta:+.3f}")
    
    # Поздние слои (28-35)
    late = [r for r in results if r[0] >= 28]
    late_acc = np.mean([r[1] for r in late])
    late_delta = np.mean([r[3] for r in late])
    print(f"  Поздние (L28-35): REAL={late_acc:.3f}, Δ={late_delta:+.3f}")
    
    # Вывод
    if early_delta < 0.1 and mid_delta > 0.5 and late_delta > 0.5:
        print("\n✅ Цифры появляются в фазе кипения (L13-27), стабилизируются в кристаллизации")
    elif early_delta > 0.5:
        print("\n⚠️ Цифры читаемы уже в ранних слоях — опровергает всю предыдущую картину!")
    elif late_delta > 0.5 and mid_delta < 0.3:
        print("\n✅ Цифры возникают только при подготовке к unembedding (кристаллизация)")
    else:
        print("\n⚠️ Нелинейная структура распределена по всем слоям")

if __name__ == "__main__":
    main()