<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl.py -b 256 -a resnet50 -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/baseline_4gpu 
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_pcb.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_4gpu --alpha 0.2
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_pcb_softlabel.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_softlabel_uselocaldist_4gpu --theta 0.8 --soft-epoch 10  --use-local-dist
=======
CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl_ss_camera.py -b 256 -a resnet50 -d market1501 --iters 400 --momentum 0.1 --eps 0.4 --num-instances 16 --logs-dir logs/ss_camera_avg_jicheng_0.4lr --pooling-type avg
CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl_ss_camera.py -b 256 -a resnet50 -d market1501 --iters 400 --momentum 0.1 --eps 0.4 --num-instances 16 --logs-dir logs/ss_camera_gem_jicheng_0.4lr

# CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl_self_supervised.py -b 256 -a resnet_ibn50a -d market1501 --iters 200 --momentum 0.1 --eps 0.4 --num-instances 16
# CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl_self_supervised.py -b 256 -a resnet_ibn50a -d market1501 --iters 400 --momentum 0.1 --eps 0.4 --num-instances 16 --pooling-type gem --use-hard
# CUDA_VISIBLE_DEVICES=0,1 python examples/cluster_contrast_train_usl_self_supervised.py -b 256 -a resnet_ibn50a -d dukemtmcreid --iters 400 --momentum 0.1 --eps 0.6 --num-instances 16 --pooling-type gem --use-hard 
>>>>>>> f6359990a4326375f23c3fd654df3fc6dcc9c579

# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_pcb_softlabel.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_softlabel_uselocallabel_4gpu --theta 0.8 --soft-epoch 10 --use-local-label
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_pcb_softlabel.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_softlabelhard_uselocallabel_4gpu --theta 0.8 --soft-epoch 10 --use-hard-label --use-local-label
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_pcb_softlabel.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_softlabel_4gpu --theta 0.8 --soft-epoch 10 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/cluster_contrast_train_usl_pcb_softlabel.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_softlabelhard_4gpu --theta 0.8 --soft-epoch 10 --use-hard-label
# CUDA_VISIBLE_DEVICES=3 python examples/cluster_contrast_train_usl_pcb_softlabel.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_softlabel_nolocallabel_uselocaldist --theta 0.8 --soft-epoch 10 --use-local-dist
# CUDA_VISIBLE_DEVICES=3 python examples/cluster_contrast_train_usl_pcb_softlabel.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_softlabel_uselocaldist_maingloballabel --theta 0.8 --soft-epoch 10 --use-local-dist --use-local-label 
# CUDA_VISIBLE_DEVICES=3 python examples/cluster_contrast_train_usl_pcb_softlabel.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_softlabelhard_uselocaldist_maingloballabel --theta 0.8 --soft-epoch 10 --use-local-dist --use-local-label --use-hard-label
# CUDA_VISIBLE_DEVICES=3 python examples/cluster_contrast_train_usl_pcb_softlabel.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_softlabel_nolocallabel_alpha0.4 --theta 0.8 --soft-epoch 10 --theta 0.5
# CUDA_VISIBLE_DEVICES=3 python examples/cluster_contrast_train_usl_pcb.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_1x3090_alpha0.4 --alpha 0.4
# CUDA_VISIBLE_DEVICES=3 python examples/cluster_contrast_train_usl.py -b 64 -a resnet50 -d market1501 --iters 800 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/baseline_1x3090_4x64 --eval-step 5 --lr 0.0000875
# CUDA_VISIBLE_DEVICES=3 python examples/cluster_contrast_train_usl_pcb_softlabel.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_softlabel_hard --theta 0.8 --soft-epoch 10
# CUDA_VISIBLE_DEVICES=3 python examples/cluster_contrast_train_usl_pcb_softlabel.py -b 256 -a pcb -d market1501 --iters 200 --momentum 0.1 --eps 0.6 --num-instances 16 --logs-dir logs/pcb_softlabel_alpha0.4 --theta 0.8 --soft-epoch 10 --alpha 0.4
