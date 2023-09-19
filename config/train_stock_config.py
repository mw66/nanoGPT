# train a miniature character-level stock model
# good for debugging and playing on macbooks and such

out_dir = 'out_stock_char'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 100
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'stock_char'
wandb_run_name = 'mini_gpt'

dataset = 'stock_char'
gradient_accumulation_steps = 2

LOOK_BACK = 84  # one quarter
STEP_SIZE = 28  # tokens per row (price bar)
batch_size = 8  # 64
block_size = STEP_SIZE * LOOK_BACK # context of up to 252 previous bars' *tokens*

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 10 * 1000 * 1000  # with 2 GPU, batch_size 8, 1M iters will loop all 16M data points once
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
