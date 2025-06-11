datasets=("Cora" "Pubmed" "Citeseer" "Computers" "Photo" "Texas" "Cornell" "CS")

dataset="ogbn-arxiv"
device=1
date=`date +%F`
root=`pwd`
net="GCN"
hidden=128
n_layers=2
train="KD"

nohup python -u TeacherTrainer.py\
    --lr 0.0005 \
    --dataset ${dataset} --device ${device} \
    --root ${root} \
    --net ${net} \
    --hidden ${hidden} --n_layers ${n_layers} \
    > ${root}/logs/out/${date}_teacher_${dataset}_${net}_${device}.log  2>&1 &

