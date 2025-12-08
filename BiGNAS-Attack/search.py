import argparse
import logging
import os
import time

import wandb
import numpy as np
import torch


from utils import set_logging, set_seed
from dataset import CrossDomain
from model import Model, Perceptor
from train import train

def debug_cold_item_counts(split_result, cold_item_id):
    """
    Debug: 計算冷門商品在 train/valid/test 出現次數
    split_result: link_split() 的回傳結果
        - 包含 target_train_edge_index, target_valid_edge_index, target_test_edge_index
    cold_item_id: int, target domain 冷門商品 index
    """
    # === train ===
    train_edges = split_result['target_train_edge_index']
    train_count = (train_edges[1] == cold_item_id).sum().item()

    # === valid ===
    valid_edges = split_result['target_valid_edge_index']
    valid_count = (valid_edges[1] == cold_item_id).sum().item()

    # === test ===
    test_edges = split_result['target_test_edge_index']
    test_count = (test_edges[1] == cold_item_id).sum().item()

    print(f"[DEBUG] cold_item_id={cold_item_id}")
    print(f"  train 出現次數: {train_count}")
    print(f"  valid 出現次數: {valid_count}")
    print(f"  test  出現次數: {test_count}")

    return train_count, valid_count, test_count


def search(args):
    args.search = True

    wandb.init(project="BiGNAS", config=args)
    set_seed(args.seed)
    set_logging()

    logging.info(f"args: {args}")

    # === 載入資料 ===
    dataset = CrossDomain(
        root=args.root,
        categories=args.categories,
        target=args.target,
        use_source=args.use_source,
    )
    data = dataset[0]

    # === 基本統計 ===
    args.num_users = data.num_users
    args.num_source_items = data.num_source_items
    args.num_target_items = data.num_target_items
    logging.info(f"data: {data}")

    # === 模型存檔路徑 ===
    DATE_FORMAT = "%Y-%m-%d_%H:%M:%S"
    args.model_path = os.path.join(
        args.model_dir,
        f'{time.strftime(DATE_FORMAT, time.localtime())}_{"_".join(args.categories)}.pt',
    )

    # === split_result: BiGNAS 用的標準輸入格式 ===
    split_result = {
        "source_train_edge_index": data.source_link,
        "target_train_edge_index": data.target_train_edge_index,
        "target_valid_edge_index": data.target_valid_edge_index,
        "target_test_edge_index": data.target_test_edge_index,
    }
    # === 將 split_result 裡的邊輸出到檔案 ===
    import numpy as np
    os.makedirs("logs/split_edges", exist_ok=True)

    def save_edge_index(name, edge_index):
        npy_path = f"logs/split_edges/{name}.npy"
        csv_path = f"logs/split_edges/{name}.csv"
        # 存 npy
        np.save(npy_path, edge_index.cpu().numpy())
        # 存 csv（兩欄：user,item）
        np.savetxt(csv_path, edge_index.cpu().numpy().T, fmt="%d", delimiter=",")
        logging.info(f"[Search] 已輸出 {name}: {edge_index.shape}, npy={npy_path}, csv={csv_path}")

    save_edge_index("source_train_edge_index", data.source_link)
    save_edge_index("target_train_edge_index", data.target_train_edge_index)
    save_edge_index("target_valid_edge_index", data.target_valid_edge_index)
    save_edge_index("target_test_edge_index",  data.target_test_edge_index)
    # 取得 cold_item_id
    if args.cold_item_id >= 0:
        cold_item_id = args.cold_item_id
    else:
        # 自動找 target domain 中，在 train split 出現次數最少的 item
        train_items = split_result['target_train_edge_index'][1]
        unique_items, counts = train_items.unique(return_counts=True)
        cold_item_id = unique_items[counts.argmin()].item()

    # Debug 印出冷門 item 在 train/valid/test 出現次數
    debug_cold_item_counts(split_result, cold_item_id)
    args.cold_item_id = cold_item_id  # 記到 args 裡給後續訓練用

    # === 建立 BiGNAS 模型並訓練 ===
    model = Model(args)
    perceptor = Perceptor(args)
    logging.info(f"model: {model}")

    train(model, perceptor, data, args, split_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # device & mode settings
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--search", default=False, action="store_true")
    parser.add_argument("--use-meta", default=False, action="store_true")
    parser.add_argument("--use-source", default=False, action="store_true")

    # dataset settings
    parser.add_argument("--categories", type=str, nargs="+", default=["CD", "Kitchen"])
    parser.add_argument("--target", type=str, default="Kitchen")
    parser.add_argument("--root", type=str, default="data/")

    # model settings
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--model-dir", type=str, default="./save/")

    # supernet settings
    parser.add_argument("--space", type=str, nargs="+",
                        default=["gcn", "gatv2", "sage", "lightgcn", "linear"])
    parser.add_argument("--warm-up", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--T", type=int, default=1)
    parser.add_argument("--entropy", type=float, default=0.0)

    # training settings
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--eta-min", type=float, default=0.001)
    parser.add_argument("--T-max", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=15,
                        help="Top-K for hit ratio evaluation")

    # meta settings
    parser.add_argument("--meta-interval", type=int, default=50)
    parser.add_argument("--meta-num-layers", type=int, default=2)
    parser.add_argument("--meta-hidden-dim", type=int, default=32)
    parser.add_argument("--meta-batch-size", type=int, default=512)
    parser.add_argument("--conv-lr", type=float, default=1)
    parser.add_argument("--hpo-lr", type=float, default=0.01)
    parser.add_argument("--descent-step", type=int, default=10)
    parser.add_argument("--meta-op", type=str, default="gat")

    # CL超參數
    parser.add_argument('--ssl_aug_type', type=str, default='edge', choices=['edge', 'node'])
    parser.add_argument('--edge_drop_rate', type=float, default=0.2)
    parser.add_argument('--node_drop_rate', type=float, default=0.2)
    parser.add_argument('--ssl_reg', type=float, default=0.1)
    parser.add_argument('--reg', type=float, default=1e-4)
    parser.add_argument('--nce_temp', type=float, default=0.2)
    parser.add_argument('--hard_ratio', type=float, default=0.1)
    parser.add_argument('--hard_mine_interval', type=int, default=1)
    parser.add_argument('--inject_source', action='store_true')
    parser.add_argument('--inject_target', action='store_true')
    parser.add_argument('--neg_samples', type=int, default=1)

    # HardUser 參數
    parser.add_argument("--cold-item-id", type=int, default=-1,
                        help="指定 target 冷門 item；<0 自動由 train split 找最冷")


    args = parser.parse_args()
    search(args)
