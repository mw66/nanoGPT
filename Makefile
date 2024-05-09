
data:
	python data/shakespeare_char/prepare.py

train:
	python train.py config/train_shakespeare_char.py
