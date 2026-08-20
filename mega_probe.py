# -*- coding: utf-8 -*-
# mega_probe_v2.py — Final comprehensive J-space analysis with validated probes
import os, re, torch, torch.nn as nn, numpy as np, random
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from jspace import load, DEVICE, DTYPE, JProjector, _j_loop
from probe import prompt_of, result_of, OPS, corr
from probe5 import TEST, BIG

PROJ = os.environ.get("PROJ", r"E:\jspace\proj_deep.pt")
STEER = r"E:\jspace\steer.pt"

def ridge(X, Y, lam=1e-2):
    X, Y = X.float(), Y.float()
    return torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device), X.T @ Y)

class MLPProbe(nn.Module):
    def __init__(self, hidden_dim, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

# ============================================================
# 1. OPERAND ENCODING (layer-wise, sham-controlled)
# ============================================================
def analyze_operand_encoding(model, tok):
    print("=" * 70)
    print("1. OPERAND ENCODING (layer-wise, sham-controlled)")
    print("=" * 70)
    
    Xs_layers, As, Bs = [], [], []
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        full = tok.encode(p + "\n" + ans, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): hs_all = model.model(full, output_hidden_states=True).hidden_states
        n = hs_all[0].shape[1]
        for l in range(len(hs_all)):
            if l >= len(Xs_layers): Xs_layers.append([])
            Xs_layers[l].append(hs_all[l][0])
        As += [float(a)] * n; Bs += [float(b)] * n
    
    num_layers = len(Xs_layers)
    peak_layer, peak_corr = -1, 0.0
    specific_count = 0
    
    print(f"{'Layer':>6s} {'corr(A)':>8s} {'corr(B)':>8s} | {'sham_A':>7s} {'sham_B':>7s} | {'Δ_A':>6s} {'Δ_B':>6s} | spec?")
    print("-" * 75)
    
    for L in range(num_layers):
        X = torch.cat(Xs_layers[L]).float()
        A = torch.tensor(As, device=DEVICE); B = torch.tensor(Bs, device=DEVICE)
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-3)
        Xn = (X - mu) / sd
        
        W_A = ridge(Xn, (A - A.mean()).unsqueeze(1))
        W_B = ridge(Xn, (B - B.mean()).unsqueeze(1))
        rA = corr((Xn @ W_A).squeeze(), A)
        rB = corr((Xn @ W_B).squeeze(), B)
        
        torch.manual_seed(321)
        perm = torch.randperm(A.shape[0], device=DEVICE)
        W_As = ridge(Xn, (A[perm] - A[perm].mean()).unsqueeze(1))
        W_Bs = ridge(Xn, (B[perm] - B[perm].mean()).unsqueeze(1))
        sA = corr((Xn @ W_As).squeeze(), A)
        sB = corr((Xn @ W_Bs).squeeze(), B)
        
        dA, dB = rA - sA, rB - sB
        spec = (dA > 0.2) and (dB > 0.2)
        if spec: specific_count += 1
        if rA > peak_corr: peak_corr, peak_layer = rA, L
        
        marker = "✅" if spec else "  "
        if L % 4 == 0 or L == num_layers - 1:
            print(f"{L:>6d} {rA:>+8.3f} {rB:>+8.3f} | {sA:>+7.3f} {sB:>+7.3f} | {dA:>+6.3f} {dB:>+6.3f} | {marker}")
    
    print(f"\n📊 Специфичных слоёв: {specific_count}/{num_layers}")
    print(f"📊 Пик operand-corr: слой {peak_layer} (corr={peak_corr:.3f})")
    return specific_count, num_layers, peak_layer

# ============================================================
# 2. GWT CAPACITY (PCA)
# ============================================================
def analyze_gwt_capacity(model, tok):
    print("\n" + "=" * 70)
    print("2. GWT CAPACITY (PCA effective dimensionality)")
    print("=" * 70)
    
    num_layers = len(model.model.layers)
    layer_data = {l: [] for l in range(num_layers)}
    
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
        current = {}
        handles = []
        def make_hook(li):
            def hook(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                current[li] = h[0, -1, :].detach().cpu().float().numpy()
            return hook
        for l in range(num_layers):
            handles.append(model.model.layers[l].register_forward_hook(make_hook(l)))
        try:
            with torch.no_grad(): model(ids)
        finally:
            for h in handles: h.remove()
        for l in range(num_layers):
            layer_data[l].append(current[l])
    
    bottleneck_layers = []
    expansion_layers = []
    print(f"{'Layer':>6s} {'eff_dim':>8s} {'var_1st':>8s} | zone")
    print("-" * 45)
    
    for l in range(num_layers):
        X = np.array(layer_data[l])
        X_c = X - X.mean(axis=0)
        pca = PCA(n_components=min(X.shape[0], X.shape[1]))
        pca.fit(X_c)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        eff_dim = int(np.searchsorted(cumvar, 0.85) + 1)
        var_1st = pca.explained_variance_ratio_[0]
        
        if eff_dim <= 4:
            zone = "🔵 bottleneck"; bottleneck_layers.append(l)
        elif eff_dim <= 7:
            zone = "🟡 expansion"; expansion_layers.append(l)
        else:
            zone = "🔴 high-dim"
        
        if l % 4 == 0 or l == num_layers - 1 or eff_dim <= 4:
            print(f"{l:>6d} {eff_dim:>8d} {var_1st:>8.3f} | {zone}")
    
    print(f"\n📊 GWT bottleneck слоёв: {len(bottleneck_layers)}/{num_layers}")
    print(f"📊 Expansion слоёв: {len(expansion_layers)}/{num_layers}")
    if bottleneck_layers and expansion_layers:
        print(f"📊 W-форма подтверждена: bottleneck({bottleneck_layers[0]}-{bottleneck_layers[-1]}) → expansion({expansion_layers[0]}-{expansion_layers[-1]})")
    return len(bottleneck_layers), num_layers

# ============================================================
# 3. OP READOUT STABILITY
# ============================================================
def analyze_op_readout(model, tok, proj):
    print("\n" + "=" * 70)
    print("3. OPERATION READOUT STABILITY")
    print("=" * 70)
    
    d = torch.load(STEER, weights_only=True)
    W_op, muS, sdS = d["W_op"].to(DEVICE), d["mu"].to(DEVICE), d["sd"].to(DEVICE)
    
    correct = 0
    print(f"{'Prompt':28s} {'true':>4s} {'pred':>4s} | match?")
    print("-" * 55)
    for a, o, b in TEST:
        p = prompt_of(a, o, b)
        T = _j_loop(model, tok, p, proj, 48, 0)[0]
        ops_pred = []
        with torch.no_grad():
            for t in range(T.shape[0]):
                h = T[t].to(DEVICE)
                hn = ((h - muS) / sdS).unsqueeze(0).float()
                logits = (hn @ W_op.float()).squeeze()
                ops_pred.append(OPS[logits.argmax().item()])
        from collections import Counter
        pred_op = Counter(ops_pred).most_common(1)[0][0]
        match = pred_op == o
        if match: correct += 1
        print(f"{p:28s} {o:>4s} {pred_op:>4s} | {'✅' if match else '❌'}")
    
    acc = correct / len(TEST)
    print(f"\n📊 Op readout accuracy: {correct}/{len(TEST)} ({acc:.3f})")
    return acc

# ============================================================
# 4. BIFURCATION TEST (causal steering)
# ============================================================
def analyze_bifurcation(model, tok, proj):
    print("\n" + "=" * 70)
    print("4. NON-LINEAR BIFURCATION (causal steering, α sweep)")
    print("=" * 70)
    
    SRC_A, SRC_B = 42, 19
    p_sum = prompt_of(SRC_A+SRC_B, '+', 0)
    p_diff = prompt_of(SRC_A-SRC_B, '-', 0)
    T_sum = _j_loop(model, tok, p_sum, proj, 48, 0)[0]
    T_diff = _j_loop(model, tok, p_diff, proj, 48, 0)[0]
    min_len = min(T_sum.shape[0], T_diff.shape[0])
    diff_vec = (T_sum[:min_len] - T_diff[:min_len]).mean(dim=0)
    diff_vec = diff_vec / diff_vec.norm()
    
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4]
    print(f"{'Prompt':28s} {'exp':>4s} | " + " ".join(f"a={a:<4.1f}" for a in alphas))
    print("-" * 70)
    
    bifurcation_found = False
    for a, o, b in TEST:
        p = prompt_of(a, o, b); exp = result_of(a, o, b)
        row = []
        for alpha in alphas:
            ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.shape[1] > 1:
                    h = h + alpha * diff_vec.to(h.device, h.dtype).unsqueeze(0).unsqueeze(0)
                return (h,) + output[1:] if isinstance(output, tuple) else h
            handle = model.model.layers[18].register_forward_hook(hook)
            try:
                with torch.no_grad(): out = model.generate(ids, max_new_tokens=160, do_sample=False)
                txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                m = re.findall(r"=\s*(-?\d+)", txt) or re.findall(r"-?\d+", txt)
                got = int(m[-1]) if m else None
            finally:
                handle.remove()
            row.append(got)
        
        unique_vals = len(set(v for v in row if v is not None))
        if unique_vals >= 2: bifurcation_found = True
        print(f"{p:28s} {exp:>4d} | " + " ".join(f"{str(v):>6s}" for v in row))
    
    print(f"\n📊 Бифуркация обнаружена: {'✅ Да' if bifurcation_found else '❌ Нет'}")
    return bifurcation_found

# ============================================================
# 5. MLP DIGIT CRYSTALLIZATION (next-token, layer-wise)
# ============================================================
def analyze_digit_crystallization(model, tok):
    print("\n" + "=" * 70)
    print("5. MLP DIGIT CRYSTALLIZATION (next-token prediction)")
    print("=" * 70)
    
    DIGITS = set('0123456789')
    layers = [0, 8, 16, 24, 28, 32, 35]
    data = {l: {'X': [], 'Y': []} for l in layers}
    
    print("Сбор next-digit пар (64 промпта)...")
    for a, o, b in BIG:
        p = prompt_of(a, o, b)
        ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True, return_tensors="pt").input_ids.to(DEVICE)
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
    
    n_samples = len(data[layers[0]]['Y'])
    print(f"Собрано {n_samples} пар\n")
    
    def train_eval(X, Y, k=4, epochs=150, seed=42, shuffle=False):
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
    
    print(f"{'Layer':>5s} | {'REAL':>6s} {'SHAM':>6s} {'Δ':>7s} | verdict")
    print("-" * 50)
    
    results = []
    for L in layers:
        X, Y = data[L]['X'], data[L]['Y']
        acc_real = train_eval(X, Y, epochs=150)
        acc_sham = train_eval(X, Y, epochs=150, shuffle=True)
        delta = acc_real - acc_sham
        results.append((L, acc_real, acc_sham, delta))
        marker = "✅" if delta > 0.3 else ("⚠️" if delta > 0.1 else "❌")
        print(f"{L:>5d} | {acc_real:>6.3f} {acc_sham:>6.3f} {delta:>+7.3f} | {marker}")
    
    early_delta = np.mean([r[3] for r in results if r[0] <= 12]) if any(r[0] <= 12 for r in results) else 0
    late_delta = np.mean([r[3] for r in results if r[0] >= 28]) if any(r[0] >= 28 for r in results) else 0
    
    print(f"\n📊 Ранние слои (L0-12): Δ={early_delta:+.3f}")
    print(f"📊 Поздние слои (L28-35): Δ={late_delta:+.3f}")
    
    crystallization_confirmed = late_delta > 0.5 and late_delta > early_delta + 0.2
    print(f"📊 Кристаллизация подтверждена: {'✅ Да' if crystallization_confirmed else '❌ Нет'}")
    return crystallization_confirmed

# ============================================================
# MAIN
# ============================================================
def main():
    model, tok = load()
    proj = JProjector(model.config.hidden_size, "mlp", PROJ)
    
    print("🔬 MEGA PROBE v2: Comprehensive J-space Analysis")
    print(f"   Model: Qwen2.5-3B-Instruct")
    print(f"   Dataset: {len(BIG)} train, {len(TEST)} test")
    print()
    
    spec_layers, total_layers, peak_layer = analyze_operand_encoding(model, tok)
    bn_layers, _ = analyze_gwt_capacity(model, tok)
    op_acc = analyze_op_readout(model, tok, proj)
    bifurc = analyze_bifurcation(model, tok, proj)
    crystallization = analyze_digit_crystallization(model, tok)
    
    print("\n" + "=" * 70)
    print("📋 MEGA PROBE v2 SUMMARY")
    print("=" * 70)
    print(f"  Operand encoding:      {spec_layers}/{total_layers} layers specific (peak L{peak_layer})")
    print(f"  GWT bottleneck:        {bn_layers}/{total_layers} layers ≤4 dim")
    print(f"  Op readout:            {op_acc:.3f} accuracy")
    print(f"  Bifurcation:           {'✅ confirmed' if bifurc else '❌ not found'}")
    print(f"  Digit crystallization: {'✅ confirmed' if crystallization else '❌ not found'}")
    
    w_shape = (bn_layers > total_layers * 0.4) and (peak_layer > total_layers * 0.3)
    print(f"\n  W-shaped dynamics: {'✅ CONFIRMED' if w_shape else '⚠️ partial'}")
    
    if w_shape and op_acc > 0.8 and bifurc and crystallization:
        print("\n  🎉 ALL VALIDATED PROBES CONSISTENT — W-manifold hypothesis fully supported")
    else:
        print("\n  ⚠️ Some probes inconsistent — review individual sections above")

if __name__ == "__main__":
    main()