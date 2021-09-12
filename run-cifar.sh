ARCH=resnet18
EPOCHS=100
BATCH_SIZE=128


## CIFAR-10 or CIFAR-100
DATASET='cifar100' # change it to DATASET='cifar10' for cifar10 case
GPU_ID=0
DROPOUT=0.3
WD=5e-4


####################################################################
#### Training ####
####################################################################

# train an old model
CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py -a=$ARCH --dataset=$DATASET \
    -b=$BATCH_SIZE --epochs=$EPOCHS --dropout-ratio=$DROPOUT --wd=$WD

# train without PCT
CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py -a=$ARCH --dataset=$DATASET \
    -b=$BATCH_SIZE --epochs=$EPOCHS --dropout-ratio=$DROPOUT --wd=$WD

# train with PCT
for PCT_OPT in 'naive' 'FD-KL' 'FD-LM'
do 
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py -a=$ARCH --dataset=$DATASET \
        -b=$BATCH_SIZE --epochs=$EPOCHS --dropout-ratio=$DROPOUT --wd=$WD \
        --pct-opt=$PCT_OPT \
        --old-arch=$ARCH \
        --old-model-path=checkpoints/$DATASET-$ARCH-PCT:None-exp01/model_best.pth.tar
done
 

####################################################################
#### Evaluation ####
####################################################################

# evaluation for training without PCT
CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py \
    -a=$ARCH --dataset=$DATASET --dropout-ratio=$DROPOUT -e \
    --resume=checkpoints/$DATASET-$ARCH-PCT:None-exp02/model_best.pth.tar \
    --old-arch=$ARCH \
    --old-model-path=checkpoints/$DATASET-$ARCH-PCT:None-exp01/model_best.pth.tar

# evaluation for PCT
for PCT_OPT in 'naive' 'FD-KL' 'FD-LM'
do 
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py \
        -a=$ARCH --dataset=$DATASET --dropout-ratio=$DROPOUT -e \
        --pct-opt=$PCT_OPT \
        --resume=checkpoints/$DATASET-$ARCH-PCT:$PCT_OPT-exp01/model_best.pth.tar \
        --old-arch=$ARCH \
        --old-model-path=checkpoints/$DATASET-$ARCH-PCT:None-exp01/model_best.pth.tar
done