# EDSR baseline model (x2)

# Training EDSR baseline as student model
#python main.py --model EDSR --scale 2 --patch_size 96 --save edsr_baseline_x2_KD --reset --data_test Set5 --pre_train download --save_result

# Test student model on the testset
#python main.py --data_test DIV2K --data_range 801-900 --scale 2 --save edsr_baseline_x2_test --pre_train download --pre_train_s ../experiment/edsr_baseline_x2_KD/model/model_best.pt  --test_only  --save_results --self_ensemble


# EDSR in the paper (x2)

# Training EDSR as student model
#python main.py --model EDSR --data_test Set5+DIV2K --scale 2 --save edsr_x2_KD --n_resblocks_s 32 --n_feats_s 256 --res_scale_s 0.1 --pre_train download --reset --save_result --batch_size 4 --test_every 4000

# Test student model on the testset

#python main.py --data_test DIV2K --data_range 801-900 --scale 2 --save edsr_x2_KD_test --n_resblocks_s 32 --n_feats_s 256 --res_scale_s 0.1 --pre_train download --pre_train_s ../experiment/edsr_x2_KD/model/model_best.pt --test_only --chop --save_results --self_ensemble --reset

