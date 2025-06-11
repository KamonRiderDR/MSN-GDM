datasets=("Cora" "Pubmed" "Citeseer" "Computers" "Photo" "Texas" "Cornell" "CS")

runs=1
times=1
dataset="Pubmed"
device=3
date=`date +%F`
root=`pwd`
net="GCN"
student_model="SpecMLP"
distill_type="SpecMLP"
hidden=128 
n_layers=3
dist_metrics="ms_cos"
train="train"
stu_hidden=64
dropout=0.2
weight_decay=0.0005          # 0.0005, 0
lr=0.0005                 # 0.0005, 0.001, 0.005,0.01
patience=1000           # 200
epochs=3000

nohup python -u StudentTrainer.py\
    --hidden ${hidden} \
    --stu_layers 3 \
    --n_layers ${n_layers}\
    --stu_hidden ${stu_hidden} \
    --times ${times} \
    --runs ${runs}\
    --lr ${lr} \
    --dataset ${dataset} \
    --device ${device} \
    --root ${root} \
    --net ${net} \
    --student_model ${student_model}\
    --distill_type ${distill_type} \
    --dist_metrics ${dist_metrics} \
    --train ${train} \
    --dropout ${dropout} \
    --weight_decay ${weight_decay} \
    --patience ${patience} \
    --epochs ${epochs} \
    > ${root}/logs/out/${date}_${dataset}_${net}_${student_model}_${device}.log  2>&1 &