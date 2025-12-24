# SGL → BiGNAS 實驗流程

## Step 1. 設定 SGL 訓練參數
SGL 的訓練設定檔：

```text
/BiGNAS/SGL-BiGNAS-new/SGL-Torch/conf/SGL.ini
```

可在此檔案中調整參數，例如：

* `epochs` (目前預設50)
* `learning_rate`
* `batch_size`
* 其他 SGL 訓練相關超參數

---

## Step 2. 執行 SGL 訓練

在 `SGL-Torch` 目錄下，使用以下指令訓練 SGL：

```bash
python main.py --recommender=SGL --dataset=all_data --aug_type=ED --reg=1e-4 --n_layers=3 --ssl_reg=0.5 --ssl_ratio=0.1 --ssl_temp=0.2
```

此步驟會產生：

* SGL 訓練完成後的 **user embeddings** 及 **E_add_source.npy**

---

## Step 3. 執行 BiGNAS（使用 SGL 輸出）

### Step 3.a 複製 SGL 預訓練 User Embedding

將 SGL 訓練完成後產生的 user embedding：

```text
/SGL-Torch/dataset/all_data/pretrain-embeddings/SGL/final/user_embeddings_final.npy
```

複製到 BiGNAS 預期讀取的位置：

```text
/BiGNAS-Attack/logs/sgl_emb/user_embeddings_final.npy
```
---

### Step 3.b 複製 Hard Item Split（加邊資訊）

將 SGL 產生的 source domain 加邊檔案：

```text
/SGL-Torch/logs/hard_item_split_v2/E_add_source.npy
```

複製到 BiGNAS 對應位置：

```text
/BiGNAS-Attack/logs/hard_item_split_v2/E_add_source.npy
```

此檔案會被 BiGNAS 用於後續 **hard item / hard user attack 的加邊操作**。

---
# SGL-BiGNAS-loss1
