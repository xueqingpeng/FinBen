from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="TheFinAI/OCR_Task",
    repo_type="dataset",              # 关键：数据集一定要写 dataset
    local_dir="/gpfs/radev/project/xu_hua/xp83/OCR_Task",         # 下载保存位置
    revision="main",                  # 或某个 tag/commit
    allow_patterns=None,              # 只下子集时可填 ["**.parquet", "train/**"]
    ignore_patterns=None,             # 排除某些大文件
    max_workers=4,                    # 并发数，视网络/CPU调整
    local_dir_use_symlinks=False      # 需要可移动/打包时建议 False（避免硬链接）
)

print("saved to:", local_path)
