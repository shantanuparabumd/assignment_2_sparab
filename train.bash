# python -m code.train_model --type 'point' --batch_size 4
# python -m code.train_model --type 'mesh' --batch_size 4

python -m code.train_model --type 'point' --batch_size 8
# python -m code.train_model --type 'mesh' --batch_size 8

# python -m code.train_model --type 'point' --batch_size 16
# python -m code.train_model --type 'mesh' --batch_size 16

python -m code.train_model --type 'point' --batch_size 32 --num_workers 12 --save_freq 100 --max_iter 1000
python -m code.train_model --type 'mesh' --batch_size 32 --num_workers 12 --save_freq 100 --max_iter 1000


# python -m code.train_model --type 'vox' --batch_size 4