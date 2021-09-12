## ImageNet
DATASET=imagenet
DATA_DIR='~/dataset/imagenet'

ARCH=resnet18
EPOCHS=100
BATCH_SIZE=512


####################################################################
#### Training ####
####################################################################

# train without PCT
python3 main.py -a=$ARCH --dataset=$DATASET --data-dir=$DATA_DIR \
    -b=$BATCH_SIZE --epochs=$EPOCHS -p=200

# train with PCT
for PCT_OPT in 'naive' 'FD-KL' 'FD-LM'
do 
    python3 main.py -a=$ARCH --dataset=$DATASET --data-dir=$DATA_DIR \
        -b=$BATCH_SIZE --epochs=$EPOCHS -p=200 \
        --pct-opt=$PCT_OPT \
        --old-arch=$ARCH --old-pretrained 
done


####################################################################
#### Evaluation ####
####################################################################

for PCT_OPT in 'None' 'naive' 'FD-KL' 'FD-LM'
do 
    python3 main.py -a=$ARCH --dataset=$DATASET --data-dir=$DATA_DIR \
        -b=$BATCH_SIZE -e \
        --pct-opt=$PCT_OPT \
        --resume=checkpoints/$DATASET-$ARCH-PCT:$PCT_OPT-exp01/model_best.pth.tar \
        --old-arch=$ARCH --old-pretrained 
done
