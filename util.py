import datetime
import glob
import random
import numpy as np
from nose.tools import *
import pdb

def now():
  return datetime.datetime.utcnow() # now is a datetime object
# ts = int(now().timestamp())

# poor man's data loader, read everything into memory
data_dir = "/data/stock/label/chatgpt/"

trns = []
vals = []
tsts = []
FIRST_MARKER = 0  # of each row (price bar)

def read_data():
  global FIRST_MARKER
  fns = glob.glob(data_dir + "*.bin")
  print(len(fns), " files")
  ONE_YEAR_DAY = 252
  ONE_YEAR_TOKENS = ONE_YEAR_DAY * STEP_SIZE
  assert_equal(block_size % STEP_SIZE, 0)
  for fn in fns:
    arr = np.fromfile(fn, dtype=np.uint16)
    if FIRST_MARKER == 0:
      FIRST_MARKER = arr[0]
    else:
      assert_equal(FIRST_MARKER, arr[0])
    # last 2 year, as val, tst data
    assert_equal(len(arr) % STEP_SIZE, 0)
    trns.append(arr[                                   : -2 * ONE_YEAR_TOKENS])
    vals.append(arr[  -2 * ONE_YEAR_TOKENS - block_size: -1 * ONE_YEAR_TOKENS])
    tsts.append(arr[  -1 * ONE_YEAR_TOKENS - block_size:                     ])


# shuffle the symbols
def shuffle_data():
  random.shuffle(trns)
  random.shuffle(vals)
  random.shuffle(tsts)


def get_batch(split):
    data = trns if split == 'train' else vals
    # random pick one symbol
    data = random.choice(data)
    ix = torch.randint( (len(data) - block_size) // STEP_SIZE, (batch_size,))
    x = torch.stack([torch.from_numpy((data[STEP_SIZE * (i  ) : STEP_SIZE * (i  )+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[STEP_SIZE * (i+1) : STEP_SIZE * (i+1)+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    assert_equal(x.shape, y.shape)
    assert_equal((x[:, 0] == FIRST_MARKER).sum(), batch_size)  # make sure we always take the arr at the row (bar) boundary
    assert_equal((y[:, 0] == FIRST_MARKER).sum(), batch_size)
    return x, y  # x.shape torch.Size([64, 256]) y.shape torch.Size([64, 256])

def decode_stock(pred):
  # 3773
  pdb.set_trace()
