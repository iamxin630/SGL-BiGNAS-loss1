import os
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

def find_hard_items_and_export_verbose(
    model,
    groupA_ids,
    hard_user_ids,
    num_users,
    num_source_items,
    num_target_items,
    k_source,
    save_dir,
    preview_top_users
):
    """
        ✔ Source domain：B 喜歡但 Hard User 不喜歡 (mean_B - mean_H)
        ❌ Target domain：全部移除
    """

    os.makedirs(save_dir, exist_ok=True)
    device = model.device
    model.lightgcn.eval()

    # === Step 1. 匯出 embedding ===
    with torch.no_grad():
        uemb, iemb = model.lightgcn._forward_gcn(model.lightgcn.norm_adj)
        uemb = F.normalize(uemb, dim=1)
        iemb = F.normalize(iemb, dim=1)
    print("=" * 80)
    print(f"[1] Embedding ready: user={tuple(uemb.shape)}, item={tuple(iemb.shape)}")

    total_items = iemb.size(0)
    B_users = torch.tensor(groupA_ids, dtype=torch.long, device=device)
    hardU = torch.tensor(hard_user_ids, dtype=torch.long, device=device)

        # === Step 2. 計算 Group B / Hard Users 的預測分數 ===
    with torch.no_grad():
        # scores_B: [#B, num_items]
        scores_B = model.lightgcn.predict(B_users)
        # mean_B: [num_items]（對所有 Group B 取平均）
        mean_B = scores_B.mean(dim=0)   # 不要 keepdim，後面比較好用

        # scores_H: [#H, num_items]
        scores_H = model.lightgcn.predict(hardU)

    print("=" * 80)
    print("[2] 已取得 Group B / Hard Users 的預測分數")

    # === Step 2.1 只取 Source domain 的 item（用真實 item id 範圍）===
    SOURCE_MIN = 2809
    SOURCE_MAX = 31061

    # global item id（= predict 出來的第幾個欄位 index）
    source_item_ids = torch.arange(SOURCE_MIN, SOURCE_MAX + 1, device=device)

    # mean_B 在 source domain 上的分數: [num_src_items]
    mean_B_src = mean_B[source_item_ids]               # shape: [num_src_items]

    # Hard users 在 source domain 上的分數: [#H, num_src_items]
    scores_H_src = scores_H[:, source_item_ids]        # shape: [#H, num_src_items]

    # Debug：確認長度
    print(">>> DEBUG: num_source_items(from range) =", len(source_item_ids))
    print(">>> DEBUG: source_item_ids[0:5] =", source_item_ids[:5].tolist())
    print(">>> DEBUG: source_item_ids[-5:] =", source_item_ids[-5:].tolist())
    print(">>> DEBUG: mean_B_src shape =", mean_B_src.shape)
    print(">>> DEBUG: scores_H_src shape =", scores_H_src.shape)

    # === Step 2.2 針對「每個 hard user」計算 Δ(u, i) = mean_B(i) - score_H(u, i) ===
    # mean_B_src:        [num_src_items]
    # scores_H_src:      [#H, num_src_items]
    # → diff:            [#H, num_src_items]
    diff = mean_B_src.unsqueeze(0) - scores_H_src
    diff = torch.nan_to_num(diff, nan=0.0)

    print("=" * 80)
    print("[2.2] 已計算每個 hard user 的 Δ(u, i) = mean_B(i) - score_H(u, i)")

    # === Step 2.3 對每個 hard user 個別做 top-k ===
    num_hard = diff.size(0)
    k_eff = min(k_source, diff.size(1))

    # vals: [#H, k_eff]  每個 hard user 的 top-k 差距值
    # idxs: [#H, k_eff]  對應在 source_item_ids 裡的 index（0 ~ num_src_items-1）
    vals, idxs = torch.topk(diff, k=k_eff, dim=1)

    # 對應回真正的 global item id: [#H, k_eff]
    selected_items_per_user = source_item_ids[idxs]    # shape: [#H, k_eff]

    # Debug：印前幾個 hard user 的 top-k item
    print("=" * 80)
    print(f"[Source] 每個 Hard User 各自 Δ 最大的 {k_eff} 個 source items (前 3 位 Hard User)：")
    for u_row in range(min(3, num_hard)):
        uid = hard_user_ids[u_row]
        items = selected_items_per_user[u_row].cpu().tolist()
        print(f"  Hard User {uid}: items = {items}")

    # === Step 3. 加邊 ===
    print("\n=== All added source edges ===")
    all_source_edges = []
    preview_log = []

    num_hard = len(hard_user_ids)
    k_eff = selected_items_per_user.size(1)

    for row, uid in enumerate(hard_user_ids):
        for j in range(k_eff):
            iid_global = int(selected_items_per_user[row, j].item())
            all_source_edges.append((uid, iid_global))

            print(f"  + user {uid}  ->  item {iid_global}")

            if len(preview_log) < preview_top_users * k_source:
                local_src_idx = int(idxs[row, j].item())  # 在 source_item_ids 裡的位置
                hard_score = float(scores_H_src[row, local_src_idx].item())
                b_mean = float(mean_B_src[local_src_idx].item())
                diff_val = float(vals[row, j].item())
                preview_log.append((uid, iid_global, hard_score, b_mean, diff_val))

    # === Step 4. 輸出為 tensor / csv ===
    def make_edge_tensor(edge_list):
        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t()

    E_add_source = make_edge_tensor(all_source_edges)
    np.save(os.path.join(save_dir, "E_add_source.npy"), E_add_source.cpu().numpy())

    print("=" * 80)
    print(f"[3] 完成：Hard Users = {len(hard_user_ids)}")
    print(f"    Source 假邊數量：{E_add_source.size(1)} 條")

    # === 預覽前幾筆 ===
    print("=" * 80)
    print(f"[4] 🔍 Hard User 加邊預覽 (前 {preview_top_users} 位)")
    print(f"{'User':>6} | {'Item':>6} | {'HardScore':>10} | {'B_Mean':>10} | {'Δ':>10}")
    print("-" * 60)
    for uid, iid, sc_h, sc_b, diff in preview_log:
        print(f"{uid:>6d} | {iid:>6d} | {sc_h:>10.6f} | {sc_b:>10.6f} | {diff:>10.6f}")

    # === CSV ===
    src_df = pd.DataFrame(E_add_source.cpu().numpy().T, columns=["user_id", "item_id"])
    src_df.to_csv(os.path.join(save_dir, "E_add_source.csv"), index=False)

    print("=" * 80)
    print("[5] 輸出完成：source 假邊 .npy + CSV 版")

    return E_add_source, src_df
