# follow the PNDM instruction to download ckpt and fid stats
# with the following configuration, fid should less 7, (6.26 tested on my machine)

model_path=~/projects/PNDM_copy/ckpts/ddim_celeba.ckpt
fid_train_celeba=~/projects/PNDM_copy/ckpts/fid_celeba_train.npz

num_step=10
order=3
path=temp/quadei_celeba_$order\_$num_step
mpiexec -np 4 python main.py --runner sample --config ddim_celeba.yml --model_path $model_path --image_path $path --sample_speed $num_step --ei_order $order --ei_str quad --last_step
pytorch-fid $path $fid_train_celeba 2>&1 | tee $path/fid.txt
