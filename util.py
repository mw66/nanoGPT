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
FIRST_MARKER = -1  # of each row (price bar)

def read_data():
  global FIRST_MARKER
  fns = glob.glob(data_dir + "*.bin")
  print(len(fns), " files")
  ONE_YEAR_DAY = 252
  ONE_YEAR_TOKENS = ONE_YEAR_DAY * STEP_SIZE
  assert_equal(block_size % STEP_SIZE, 0)
  for fn in fns:
    arr = np.fromfile(fn, dtype=np.uint16)
    if FIRST_MARKER == -1:
      FIRST_MARKER = arr[-1]
    else:
      assert_equal(FIRST_MARKER, arr[-1])
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
    data = random.choice(data)  # random pick one symbol

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i  :i  +block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# return data of all the symbols for the same day
def get_batch_tst(pred_day_i=-1):
  xs = []
  ys = []
  for data in tsts:  # each stock
    # always translate to positive array index, to make array slice indices always work as expected
    if pred_day_i < 0:
      pred_day_i += len(data) // STEP_SIZE
    x_start = STEP_SIZE * (pred_day_i+1) - block_size  # to make the convention that: pred_day_i=-1 for live day work
    y_start = STEP_SIZE + x_start
    x_end = x_start + block_size
    y_end = y_start + block_size
    xs.append(torch.from_numpy(data[x_start: x_end].astype(np.int64)))
    ys.append(torch.from_numpy(data[y_start: y_end].astype(np.int64)))

  return get_batch_xy(xs, ys)


def get_batch_with_step(split):
    data = trns if split == 'train' else vals
    data = random.choice(data)  # random pick one symbol

    # we only train the close price, but need in two steps (dollar, cent)
    bs = batch_size // 2
    assert_equal(batch_size, bs * 2)  # must be an even number

    ix = torch.randint(1, (len(data) - block_size) // STEP_SIZE, (bs,))
    xs = []
    ys = []
    for i in ix:
      x_start = STEP_SIZE * (i  )
      y_start = STEP_SIZE * (i+1)
      # seen previous bar's dollar, predict the next cent
      xs.append(torch.from_numpy(data[x_start-1: x_start-1+block_size].astype(np.int64)))  # cent
      ys.append(torch.from_numpy(data[y_start-1: y_start-1+block_size].astype(np.int64)))

      # seen full previous bar, predict the next dollar
      xs.append(torch.from_numpy(data[x_start  : x_start+  block_size].astype(np.int64)))  # dollar
      ys.append(torch.from_numpy(data[y_start  : y_start+  block_size].astype(np.int64)))


def get_batch_xy(xs, ys):
    x = torch.stack(xs)
    y = torch.stack(ys)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    assert_equal((x[:, -1] == FIRST_MARKER).sum(), len(x))  # make sure we always take the arr at the row (bar) boundary
    assert_equal((y[:, -1] == FIRST_MARKER).sum(), len(y))
    assert_equal(x.shape, y.shape)  # not for live data
    return x, y  # x.shape torch.Size([64, 256]) y.shape torch.Size([64, 256])


def decode_stock(pred):
  # 3773
  pdb.set_trace()
