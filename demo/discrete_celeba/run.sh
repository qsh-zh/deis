# follow the PNDM instruction to download ckpt and fid stats
# with the following configuration, fid should less 7, (6.35 tested on my machine)

model_path=~/projects/PNDM_copy/ckpts/ddim_celeba.ckpt
fid_train_celeba=~/projects/PNDM_copy/ckpts/fid_celeba_train.npz

num_step=10
path=temp/deis_celeba
mpiexec -np 4 python main.py --runner sample --config ddim_celeba.yml --model_path $model_path --image_path $path --sample_speed $num_step
pytorch-fid $path $fid_train_celeba 2>&1 | tee $path/fid.txt
