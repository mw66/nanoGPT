CUDA_VISIBLE_DEVICES=0,1 \
TORCHDYNAMO_DISABLE=1 \
TORCHDYNAMO_SUPPRESS_ERRORS=True \
OMP_NUM_THREADS=2 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096 \
	torchrun --standalone --nproc_per_node gpu \
	train_stock.py config/train_stock_config.py
