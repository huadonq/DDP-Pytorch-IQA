# DataParallel mode train.sh. config.distributed=False
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ../../tools/test.py --work-dir ./
# DistributedDataParallel mode train.sh. config.distributed=True
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_addr 127.0.1.76 --master_port 30076 ../../tools/test.py --work-dir ./