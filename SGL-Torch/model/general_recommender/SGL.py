# Paper: Self-supervised Graph Learning for Recommendation
# Author: Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie
# Reference: https://github.com/wujcan/SGL-Torch
# """

# __author__ = "Jiancan Wu"
# __email__ = "wujcan@gmail.com"

# __all__ = ["SGL"]

import torch
from torch.serialization import save
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import get_initializer
from util.common import Reduction
from data import PointwiseSamplerV2, PairwiseSamplerV2
import numpy as np
from time import time
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix, ensureDir
from util.pytorch import sp_mat_to_sp_tensor
from reckit import randint_choice
from analyze_hard_items import find_hard_items_and_export_verbose

class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.dropout = nn.Dropout(0.1)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        # # weight initialization
        # self.reset_parameters()

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain:
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding).cuda()
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding).cuda()
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)
        else:
            init = get_initializer(init_method)
            init(self.user_embeddings.weight)
            init(self.item_embeddings.weight)

    def forward(self, sub_graph1, sub_graph2, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)

        # Normalize embeddings learnt from sub-graph to construct SSL loss
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        neg_item_embs = F.embedding(neg_items, item_embeddings)
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        sup_pos_ratings = inner_product(user_embs, item_embs)       # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)   # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings              # [batch_size]

        pos_ratings_user = inner_product(user_embs1, user_embs2)    # [batch_size]
        pos_ratings_item = inner_product(item_embs1, item_embs2)    # [batch_size]
        tot_ratings_user = torch.matmul(user_embs1, 
                                        torch.transpose(user_embeddings2, 0, 1))        # [batch_size, num_users]
        tot_ratings_item = torch.matmul(item_embs1, 
                                        torch.transpose(item_embeddings2, 0, 1))        # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]                  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]                  # [batch_size, num_users]

        return sup_logits, ssl_logits_user, ssl_logits_item, user_embeddings1

    def _forward_gcn(self, norm_adj):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj)


class SGL(AbstractRecommender):
    def __init__(self, config):
        super(SGL, self).__init__(config)

        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        # General hyper-parameters
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.learner = config["learner"]
        self.lr = config['lr']
        self.param_init = config["param_init"]

        # Hyper-parameters for GCN
        self.n_layers = config['n_layers']

        # Hyper-parameters for SSL
        self.ssl_aug_type = config["aug_type"].lower()
        assert self.ssl_aug_type in ['nd','ed', 'rw']
        self.ssl_reg = config["ssl_reg"]
        self.ssl_ratio = config["ssl_ratio"]
        self.ssl_mode = config["ssl_mode"]
        self.ssl_temp = config["ssl_temp"]

        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = '#layers=%d-reg=%.0e' % (
            self.n_layers,
            self.reg
        )
        self.model_str += '/ratio=%.1f-mode=%s-temp=%.2f-reg=%.0e' % (
            self.ssl_ratio,
            self.ssl_mode,
            self.ssl_temp,
            self.ssl_reg
        )
        self.pretrain_flag = config["pretrain_flag"]
        if self.pretrain_flag:
            self.epochs = 0
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % (
                self.dataset_name, 
                self.model_name,
                self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % (
                self.dataset_name, 
                self.model_name,
                self.n_layers,)
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)
            print("[DEBUG] tmp_model_dir =", self.tmp_model_dir)
            print("[DEBUG] save_dir =", self.save_dir)

        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items

        # === Debug: check isolated users/items ===
        csr = self.dataset.train_data.to_csr_matrix()
        u_freq = csr.sum(1)
        i_freq = csr.sum(0)

        print("====== DEGREE CHECK ======")
        print("Zero-degree users:", np.sum(u_freq.A.flatten() == 0))
        print("Zero-degree items:", np.sum(i_freq.A.flatten() == 0))
        print("===========================")
        zero_user = np.sum(u_freq.A.flatten() == 0)
        zero_item = np.sum(i_freq.A.flatten() == 0)

        print("Zero user rate:", zero_user / self.num_users)
        print("Zero item rate:", zero_item / self.num_items)


        # === 指定 Group A: 買過 target item 的 users ===
        group_a_ids = [50, 99, 119, 191, 260, 550, 735, 946, 1175, 1615]
        self.user_group_tensor = torch.zeros(self.num_users, dtype=torch.long)
        self.user_group_tensor[group_a_ids] = 1  # 1: Group A, 0: 其他

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)
        if self.pretrain_flag:
            self.lightgcn.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        else:
            self.lightgcn.reset_parameters(init_method=self.param_init)
        # ============================================================
        # [NEW] 1. 個性化溫度預計算 (Group-based)
        # ============================================================
        # === Personalized Temperature (tau_u) ===
        
        # 1. 取得 Group A 與 Group B
        # group_a_ids 已在上方定義
        all_users = np.arange(self.num_users)
        group_b_ids = np.array([u for u in all_users if u not in group_a_ids])
        
        # 2. 計算 Degree (冷門程度)
        users_items = self.dataset.train_data.to_user_item_pairs()
        user_degrees = np.bincount(users_items[:, 0], minlength=self.num_users)
        degrees_b = user_degrees[group_b_ids]
        
        # 3. 計算與 Group A 的不相似度 (Dissimilarity)
        self.lightgcn.eval()
        with torch.no_grad():
            # 使用當前 (初始或預訓練) 的 embedding
            user_emb = self.lightgcn.user_embeddings.weight
            user_emb = F.normalize(user_emb, dim=1)
            
            emb_a = user_emb[group_a_ids]
            emb_b = user_emb[group_b_ids]
            
            # 計算 B 中每個用戶與 A 中所有用戶的最大相似度
            # [len(B), len(A)]
            sim_matrix = torch.matmul(emb_b, emb_a.T) 
            max_sim_b, _ = torch.max(sim_matrix, dim=1)
            dissimilarity_b = (1 - max_sim_b).cpu().numpy()
            
        # 4. 找出「最冷門」且「最不像」的交集，佔 Group B 的 10%
        # degree 越小越好 (冷門)，dissimilarity 越大越好 (不像)
        rank_degree = np.argsort(np.argsort(degrees_b))       # 0-indexed rank, 越小越冷門
        rank_dissim = np.argsort(np.argsort(-dissimilarity_b)) # 0-indexed rank, 越小越不像
        
        # 取兩者排名的最大值 (代表要進入前 N 名，必須兩者都在前 N 名內)
        combined_rank = np.maximum(rank_degree, rank_dissim)
        
        # 找出 combined_rank 最小的前 10% (這就是兩者 Top-N 的交集)
        num_b = len(group_b_ids)
        k = int(num_b * 0.1)
        target_indices_in_b = np.argsort(combined_rank)[:k]
        target_user_ids = group_b_ids[target_indices_in_b]
        
        print(f"[DEBUG] Selected {len(target_user_ids)} users from Group B as 'Cold & Dissimilar' targets.")

        # 5. 設定溫度
        # 預設為 0.5 (Head Users)
        self.user_temps = np.full(self.num_users, 0.5, dtype=np.float32)
        
        # [動作] 針對「最冷門且最不像 Group A」的 Group B 用戶給予低溫度 (0.1)
        self.user_temps[target_user_ids] = 0.1
        
        # [動作] 針對目標群體 Group A 同樣給予低溫度 (0.1)
        self.user_temps[group_a_ids] = 0.1
        
        self.user_temps = torch.tensor(self.user_temps, dtype=torch.float32).to(self.device)

        # self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    @timer
    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        if is_subgraph and self.ssl_ratio > 0:
            if aug_type == 'nd':
                drop_user_idx = randint_choice(self.num_users, size=self.num_users * self.ssl_ratio, replace=False)
                drop_item_idx = randint_choice(self.num_items, size=self.num_items * self.ssl_ratio, replace=False)
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(users_np, dtype=np.float32), (users_np, items_np)), 
                    shape=(self.num_users, self.num_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep+self.num_users)), shape=(n_nodes, n_nodes))
            if aug_type in ['ed', 'rw']:
                keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))
        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)                    
        # self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            training_start_time = time()
            if self.ssl_aug_type in ['nd', 'ed']:
                sub_graph1 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(self.device)
                sub_graph2 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph2 = sp_mat_to_sp_tensor(sub_graph2).to(self.device)
            else:
                sub_graph1, sub_graph2 = [], []
                for _ in range(0, self.n_layers):
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph1.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph2.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))
            self.lightgcn.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                sup_logits, ssl_logits_user, ssl_logits_item, user_embs_sub1_full = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items)
                
                # BPR Loss
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                 # ============================================================
                # [MODIFIED 1] InfoNCE 改回固定溫度 (維持推薦穩定性)
                # ============================================================
                # 使用 self.ssl_temp (全域固定)
                clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
                clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
                infonce_loss = torch.sum(clogits_user + clogits_item)
                
                # ============================================================
                # [MODIFIED 2] Group Loss 改用 Softmax + Self-Tau (極化分群)
                # ============================================================
                # 取出當前 batch users 的 embedding (從 forward 傳回的 sub_graph1 embedding)
                bat_user_embs = F.embedding(bat_users, user_embs_sub1_full)
                
                # 呼叫新的 Softmax Loss
                group_loss = group_softmax_with_selftau(
                    user_embs=bat_user_embs,
                    user_ids=bat_users,
                    user_group_tensor=self.user_group_tensor,
                    user_temps=self.user_temps  # 傳入整張表的 Self-Tau
                )
                ######
                # === Total Loss ===
                alpha = 1.0     # group loss 權重
                loss = (
                    bpr_loss
                    + self.ssl_reg * infonce_loss
                    + self.reg * reg_loss
                    + alpha * group_loss
                )
                total_loss += loss
                total_bpr_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            self.logger.info(
                f"[iter {epoch} : loss = {total_loss:.4f}, "
                f"bpr = {total_bpr_loss:.4f}, reg = {total_reg_loss:.4f}, "
                f"time = {time() - training_start_time:.4f}]"
            )


        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)

        buf = "training finished without evaluation"
        self.logger.info(buf)

        # === Debug: 檢查 embedding 是否崩掉 ===
        u = self.lightgcn.user_embeddings.weight.detach().cpu().numpy()
        i = self.lightgcn.item_embeddings.weight.detach().cpu().numpy()
        print("User embedding mean/std:", u.mean(), u.std())
        print("Item embedding mean/std:", i.mean(), i.std())
        # === Debug: 檢查 embedding L2 norm 是否 normal ===
        norms = np.linalg.norm(u, axis=1)
        print("User embedding L2 norm mean/std:", norms.mean(), norms.std())

        # ★ 在這裡統一匯出「傳播後」的最終向量
        try:
            out_dir = self.export_final_embeddings(out_dir=self.save_dir)
            self.logger.info(f"export_final_embeddings done: {out_dir}")
        except Exception as e:
            self.logger.warning(f"export_final_embeddings failed: {e}")
       # ============================================================
        # 🔧 After SGL training: 分析 Hard Users 與 Hard Items
        # ============================================================
        try:
            print("\n\n================= Hard User / Hard Item Analysis =================")
            # 1️⃣ 定義 Group A
            groupA_ids = [50, 99, 119, 191, 260, 550, 735, 946, 1175, 1615]

            # 2️⃣ 計算 Hard Users（根據 cosine distance）
            with torch.no_grad():
                user_emb, _ = self.lightgcn._forward_gcn(self.lightgcn.norm_adj)
                user_emb = F.normalize(user_emb, dim=1)
                A = torch.tensor(groupA_ids, device=self.device)
                all_users = torch.arange(self.num_users, device=self.device)
                B = torch.tensor([u for u in all_users.tolist() if u not in groupA_ids], device=self.device)

                sim = torch.matmul(user_emb[B], user_emb[A].T)
                max_sim, _ = sim.max(dim=1)
                dist = 1 - max_sim
                k_hard = int(len(B) * 0.01) # top 1%
                hard_user_ids = B[torch.topk(dist, k=k_hard).indices].cpu().tolist()
                print(f"選出 {len(hard_user_ids)} 位 Hard Users（距離最大 Top10%）")

            E_add_source, top_src = find_hard_items_and_export_verbose(
                model=self,
                groupA_ids=groupA_ids,
                hard_user_ids=hard_user_ids,
                num_users=2809,
                num_source_items=28253,
                num_target_items=14274,
                k_source=5,             # 你指定的每個 Hard User 要加的 source 邊數
                save_dir="logs/hard_item_split_v2",
                preview_top_users=30
            )


            print("================= Hard Item Analysis Done =================\n\n")

        except Exception as e:
            print(f"[Warning] Hard item analysis skipped due to error: {e}")


    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()
    
    # 放在 SGL 類別內（與 evaluate_model 同層級）
    def export_final_embeddings(self, out_dir=None):
        # 1) 前向 propagation 拿最終 embedding
        self.lightgcn.eval()
        with torch.no_grad():
            user_final, item_final = self.lightgcn._forward_gcn(self.lightgcn.norm_adj)

        # 2) 轉 numpy
        user_final = user_final.detach().cpu().numpy()
        item_final = item_final.detach().cpu().numpy()

        # ================= DEBUG: 檢查 SGL 輸出的 final embedding =================
        print("\n================= DEBUG: SGL 產生的 final embedding =================")
        print("[DEBUG] SHAPE:", user_final.shape)
        print("[DEBUG] mean/std:", user_final.mean(), user_final.std())
        print("[DEBUG] L2 norm mean/std:",
            np.linalg.norm(user_final, axis=1).mean(),
            np.linalg.norm(user_final, axis=1).std())
        print("=====================================================================\n")

        # 3) 決定輸出路徑
        if out_dir is None:
            if self.save_dir is not None:
                out_dir = self.save_dir
            else:
                out_dir = self.config.data_dir + f"{self.dataset_name}/pretrain-embeddings/{self.model_name}/final/"
                ensureDir(out_dir)

        # 4) 存檔
        np.save(out_dir + 'user_embeddings_final.npy', user_final)
        np.save(out_dir + 'item_embeddings_final.npy', item_final)

        return out_dir


# === Group Softmax Loss with Self-Tau ===
def group_softmax_with_selftau(user_embs, user_ids, user_group_tensor, user_temps):
    """
    Args:
        user_embs: [batch_size, dim] (已 Normalize)
        user_ids: [batch_size]
        user_group_tensor: [num_users] (0 or 1)
        user_temps: [num_users] (0.1 ~ 0.5)
    """
    device = user_embs.device
    
    # 1. 準備數據
    user_group_tensor = user_group_tensor.to(device)
    labels = user_group_tensor[user_ids]
    # 取出對應溫度並轉為 [batch, 1] 以便廣播
    batch_temps = user_temps[user_ids].unsqueeze(1).to(device)

 # 2. 計算相似度 (Cosine Similarity)
    # [batch, batch]
    sim_matrix = torch.matmul(user_embs, user_embs.T)

    # 3. 關鍵：除以個性化溫度 (Self-Tau)
    # 冷門用戶 tau=0.1 -> sim 被放大 10 倍 -> Softmax 極度尖銳 -> 強制拉近同類
    sim_matrix = sim_matrix / batch_temps

    # 4. 數值穩定處理 (LogSumExp Trick)
    sim_matrix_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_matrix_max.detach()

    # 5. 建立 Mask
    batch_size = user_embs.shape[0]
    mask_self = torch.eye(batch_size, dtype=torch.bool, device=device) # 自己
    # [動作] 同群體拉近 (Pull): 找出與自己同屬 Group A 或 Group B 的正樣本
    # 同 Group 的人 (正樣本)
    mask_pos = labels.unsqueeze(1).eq(labels.unsqueeze(0)) & (~mask_self)
    
    # 6. 計算 Log Softmax
    # [動作] 不同群體拉遠 (Push): 分母包含所有樣本，Loss 會將不同群體的向量推開
    # 分母：所有樣本的 exp sum (除了自己)
    exp_sim = torch.exp(sim_matrix) * (~mask_self).float()
    log_prob_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # 分子 - 分母 = Log Probability
    log_prob = sim_matrix - log_prob_denom
    
    # 7. 計算 Loss (只取正樣本部分)
    # 避免除以 0 (若某人該 batch 沒同伴)
    pos_counts = mask_pos.sum(dim=1)
    pos_counts[pos_counts == 0] = 1 
    
    # SupCon Loss
    loss = - (mask_pos.float() * log_prob).sum(dim=1) / pos_counts
    return loss.mean()
