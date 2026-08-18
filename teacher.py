# -*- coding: utf-8 -*-
import os, re, random, argparse
import torch, torch.nn.functional as F
from jspace import load, DEVICE, DTYPE, MLPBridge, SparseBridge, MuSparseBridge, DeepBridge
from probe import OPS, ridge_fit, one_hot

DATA = r"E:\jspace\teacher.pt"
PROJ_MLP = r"E:\jspace\proj_mlp.pt"
PROJ_SPARSE = r"E:\jspace\proj_sparse.pt"
PROJ_DEEP = r"E:\jspace\proj_deep.pt"

def parse_ab(prompt):
    m = re.search(r"(\d+)\s*([+\-*/])\s*(\d+)", prompt)
    return int(m.group(1)), m.group(2), int(m.group(3))

def collect(model, tok, prompt, bridge=None, max_tokens=256, p_max=0.5):
    a, o, b = parse_ab(prompt)
    base, head, embed = model.model, model.lm_head, model.get_input_embeddings()
    with torch.no_grad():
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt")
        prompt_emb = embed(ids["input_ids"].to(DEVICE))
        suffix, Hs, Es, Ops, As, Bs, text = [], [], [], [], [], [], ""
        for i in range(max_tokens):
            emb = prompt_emb if not suffix else \
                  torch.cat([prompt_emb, torch.cat(suffix, 1)], 1)
            out = base(inputs_embeds=emb.to(DTYPE))
            h = out.last_hidden_state[:, -1:, :].detach().float()
            nt = int(head(h.to(DTYPE)).argmax(-1).item())
            text += tok.decode([nt])
            e_true = embed(torch.tensor([[nt]], device=DEVICE)).detach()
            Hs.append(h.squeeze(0).squeeze(0).cpu())
            Es.append(e_true.float().squeeze(0).squeeze(0).cpu())
            Ops.append(OPS.index(o)); As.append(float(a)); Bs.append(float(b))
            if nt == tok.eos_token_id or "｜User｜" in text: break
            p = p_max * i / max_tokens
            if bridge is not None and random.random() < p:
                suffix.append(bridge(h).to(DTYPE).detach())
            else:
                suffix.append(e_true)
    return (torch.stack(Hs), torch.stack(Es),
            torch.tensor(Ops), torch.tensor(As), torch.tensor(Bs))

def train(H, E, yop, yA, yB, epochs, bridge="mlp", out=None, lam_probe=0.2, mu=255.0):
    out = out or (PROJ_MLP if bridge == "mlp" else
                  PROJ_DEEP if bridge == "deep" else PROJ_SPARSE)
    with torch.no_grad():
        pA_c = (yA - yA.mean()) / yA.std(); pB_c = (yB - yB.mean()) / yB.std()
        pr_op = ridge_fit(E, one_hot(yop, len(OPS)))
        pr_A  = ridge_fit(E, pA_c.unsqueeze(1)); pr_B = ridge_fit(E, pB_c.unsqueeze(1))
    if bridge == "mu":
        net = MuSparseBridge(H.shape[1], mu=mu); net.scale.fill_(H.abs().mean().item())
    elif bridge == "sparse": net = SparseBridge(H.shape[1])
    elif bridge == "deep":   net = DeepBridge(H.shape[1])
    else:                    net = MLPBridge(H.shape[1])
    H, E = H.to(DEVICE), E.to(DEVICE)
    yop, pA_c, pB_c = yop.to(DEVICE), pA_c.to(DEVICE), pB_c.to(DEVICE)
    pr_op, pr_A, pr_B = pr_op.to(DEVICE), pr_A.to(DEVICE), pr_B.to(DEVICE)
    net = net.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lam_s, target_act, spar = 1e-3, 0.30, 0.0
    for ep in range(epochs):
        opt.zero_grad()
        if bridge == "mu":
            code = net.act(net.enc(net.compand(H))); p = net.expand(net.dec(code))
            spar = (code.abs() > 0.01).float().mean().item(); l_spar = code.abs().mean()
        elif bridge == "sparse":
            code = F.relu(net.enc(H)); p = net.dec(code)
            spar = (code > 0.01).float().mean().item(); l_spar = code.abs().mean()
        else:
            p = net(H); l_spar = torch.zeros(())
        l_fid = F.mse_loss(p, E) + (1 - F.cosine_similarity(p, E, dim=-1)).mean()
        l_probe = (F.cross_entropy(p @ pr_op, yop)
                   + F.mse_loss((p @ pr_A).squeeze(), pA_c)
                   + F.mse_loss((p @ pr_B).squeeze(), pB_c))
        loss = l_fid + lam_probe * l_probe + (lam_s * l_spar if bridge in ("mu", "sparse") else 0.0)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if bridge in ("mu", "sparse") and ep % 10 == 9:
            lam_s = min(1e-1, max(1e-4, lam_s * (1 + 0.05 * (1 if spar > target_act else -1))))
        if ep % 50 == 0:
            msg = f"epoch {ep} loss {loss.item():.4f} | fid {l_fid.item():.5f} | probe {l_probe.item():.4f}"
            if bridge in ("mu", "sparse"): msg += f" | act {spar:.3f} lam_s {lam_s:.1e}"
            print(msg)
    torch.save(net.state_dict(), out)
    print("final loss:", round(loss.item(), 4))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "dagger", "train"])
    ap.add_argument("arg", nargs="?", default="")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bridge", choices=["mlp", "sparse", "mu", "deep"], default="mlp")
    a = ap.parse_args()

    if a.cmd == "train":
        d = torch.load(DATA, weights_only=True)
        train(d["H"], d["E"], d["yop"], d["yA"], d["yB"], a.epochs, a.bridge)
    else:
        model, tok = load()
        bridge = None
        if a.cmd == "dagger":
            bridge = MLPBridge(model.config.hidden_size).to(DEVICE)
            bridge.load_state_dict(torch.load(PROJ_MLP, map_location="cpu", weights_only=True))
            bridge.eval()
        H, E, yop, yA, yB = collect(model, tok, a.arg, bridge)
        if os.path.exists(DATA):
            d = torch.load(DATA, weights_only=True)
            H, E = torch.cat([d["H"], H], 0), torch.cat([d["E"], E], 0)
            yop = torch.cat([d["yop"], yop], 0)
            yA  = torch.cat([d["yA"], yA], 0)
            yB  = torch.cat([d["yB"], yB], 0)
        torch.save({"H": H, "E": E, "yop": yop, "yA": yA, "yB": yB}, DATA)
        print("пар в датасете:", H.shape[0])