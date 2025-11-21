
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import argparse
import os

# --- 參數設定 (為確保提交成功，我們大幅降低搜索長度) ---
INITIAL_BEAM_WIDTH = 1 
SEQUENCE_LENGTH = 10  # 🔴 關鍵修改: 縮短為 10，確保能在時限內跑完
CHECKPOINT_FILE = "submission_checkpoint.csv"

def run_brute_force(device):
    print(f"🚀 Starting Brute Force Search on {device}...")
    
    # 1. 載入模型
    st_model = SentenceTransformer('/kaggle/input/sentence-t5-base-hf/sentence-t5-base', device=device)
    tokenizer = st_model.tokenizer
    
    # 2. 載入目標向量
    target_embeddings = np.load("target_embeddings.npy")
    
    # 3. 載入起始文字資料表 (作為提交檔的基底)
    df = pd.read_parquet("test_with_start_text.parquet")
    
    # 🔥 確保 'id' 是正確的欄位，並且準備 Checkpoint
    if df.index.name == 'id':
        df = df.reset_index()
    
    start_texts = df['start_text'].values
    df_checkpoint = df.copy()
    df_checkpoint['rewrite_prompt'] = "" 
    
    # 根據 device 分配工作 (單卡跑全部)
    indices = range(len(df))
    pbar = tqdm(total=len(indices), desc="Optimizing Prompts")
    
    for i in indices:
        current_text = start_texts[i]
        target_emb = torch.tensor(target_embeddings[i]).to(device)
        
        # --- Beam Search 核心邏輯 (省略內部代碼，邏輯不變) ---
        initial_input_ids = tokenizer(current_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        beams = [(initial_input_ids[0], 0.0)] 
        
        def get_embeddings(input_ids_batch):
            with torch.no_grad():
                mask = torch.ones_like(input_ids_batch).to(device)
                out = st_model({
                    'input_ids': input_ids_batch, 
                    'attention_mask': mask
                })['sentence_embedding']
            return out
        
        # 搜索循環 (SEQUENCE_LENGTH=10)
        for step in range(SEQUENCE_LENGTH):
            current_beam_width = max(1, int(INITIAL_BEAM_WIDTH - step))
            candidates = []
            for seq_ids, score in beams:
                best_token = None
                best_sim = -1
                batch_input_ids = []
                batch_tokens = []
                
                prompt_words = [" better", " more", " style", " tone", " rewrite", " rephrase", " text", " story", " poem", " code", "!", ".", " sure", " make", " change"]
                prompt_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in prompt_words]
                check_ids = list(range(1000)) + prompt_ids
                
                for token_id in check_ids: 
                    new_seq = torch.cat([seq_ids, torch.tensor([token_id]).to(device)])
                    batch_input_ids.append(new_seq)
                    batch_tokens.append(token_id)
                    if len(batch_input_ids) >= 128: 
                        batch_tensor = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                        embeddings = get_embeddings(batch_tensor)
                        sims = torch.nn.functional.cosine_similarity(embeddings, target_emb.unsqueeze(0))
                        max_val, max_idx = torch.max(sims, 0)
                        if max_val.item() > best_sim:
                            best_sim = max_val.item()
                            best_token = batch_tokens[max_idx]
                        batch_input_ids = []
                        batch_tokens = []
                
                if batch_input_ids:
                     batch_tensor = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                     embeddings = get_embeddings(batch_tensor)
                     sims = torch.nn.functional.cosine_similarity(embeddings, target_emb.unsqueeze(0))
                     max_val, max_idx = torch.max(sims, 0)
                     if max_val.item() > best_sim:
                        best_sim = max_val.item()
                        best_token = batch_tokens[max_idx]

                if best_token is not None:
                    new_seq = torch.cat([seq_ids, torch.tensor([best_token]).to(device)])
                    candidates.append((new_seq, best_sim))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:current_beam_width]
        
        final_ids = beams[0][0]
        final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
        
        # 更新結果
        df_checkpoint.iloc[i, df_checkpoint.columns.get_loc('rewrite_prompt')] = final_text
        
        # --- 中途存檔 (每5筆) ---
        if (i + 1) % 5 == 0 or (i + 1) == len(indices):
            # 確保 id 是 int
            df_checkpoint['id'] = df_checkpoint['id'].astype(int) 
            df_checkpoint[['id', 'rewrite_prompt']].to_csv(CHECKPOINT_FILE, index=False)

        pbar.update(1)
        
    pbar.close()
    
    # --- 最後存檔 ---
    print("Generating final submission.csv...")
    # 🔴 確保 id 是 int 且只輸出兩欄
    df_checkpoint['id'] = df_checkpoint['id'].astype(int)
    df_checkpoint[['id', 'rewrite_prompt']].to_csv("submission.csv", index=False)
    
    print("🎉 Submission file created: submission.csv")
    print(f"Submission Shape: {df_checkpoint[['id', 'rewrite_prompt']].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    run_brute_force(args.device)
