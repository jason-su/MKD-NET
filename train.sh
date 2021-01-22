LOGFILE=/home/jhsu/sjh/logs/CornerNet_lite_merge_class_voc-`date +%Y-%m-%d-%H-%M-%S`.log
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u train.py --iter=0 >>$LOGFILE 2>&1 &