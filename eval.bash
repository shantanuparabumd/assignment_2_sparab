# python -m code.eval_model --type 'point' --batch 4 --instance 2 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 4 --instance 4 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 4 --instance 5 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 4 --instance 8 --load_checkpoint

# python -m code.eval_model --type 'mesh' --batch 4 --instance 2 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 4 --instance 4 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 4 --instance 5 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 4 --instance 8 --load_checkpoint

# python -m code.eval_model --type 'point' --batch 8 --instance 2 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 8 --instance 4 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 8 --instance 5 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 8 --instance 8 --load_checkpoint

# python -m code.eval_model --type 'mesh' --batch 8 --instance 2 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 8 --instance 5 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 8 --instance 8 --load_checkpoint

# python -m code.eval_model --type 'point' --batch 16 --instance 2 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 16 --instance 4 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 16 --instance 5 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 16 --instance 8 --load_checkpoint

# python -m code.eval_model --type 'mesh' --batch 16 --instance 2 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 16 --instance 4 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 16 --instance 5 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 16 --instance 8 --load_checkpoint

# python -m code.eval_model --type 'point' --batch 32 --instance 2 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 32 --instance 4 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 32 --instance 5 --load_checkpoint
# python -m code.eval_model --type 'point' --batch 32 --instance 8 --load_checkpoint

# python -m code.eval_model --type 'mesh' --batch 32 --instance 2 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 32 --instance 4 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 32 --instance 5 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 32 --instance 8 --load_checkpoint


# python -m code.eval_model --type 'vox' --batch 4 --instance 2 --load_checkpoint
# python -m code.eval_model --type 'vox' --batch 4 --instance 4 --load_checkpoint
# python -m code.eval_model --type 'vox' --batch 4 --instance 5 --load_checkpoint
# python -m code.eval_model --type 'vox' --batch 4 --instance 8 --load_checkpoint



python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --num_workers 12 --step 0 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --step 1000 --load_checkpoint
python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --num_workers 12 --step 200 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --step 3000 --load_checkpoint
python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --num_workers 12 --step 400 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --step 5000 --load_checkpoint
python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --num_workers 12 --step 600 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --step 7000 --load_checkpoint
python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --num_workers 12 --step 800 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --step 9000 --load_checkpoint
# python -m code.eval_model --type 'mesh' --batch 8 --instance 4 --step 10000 --load_checkpoint
