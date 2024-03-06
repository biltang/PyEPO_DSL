#!/bin/bash
mkdir -p ./outputs/slurmlogs

# finite diff scheme
fin_diff_sch=("B" "C")

# h_exp 
h_exp=(-0.01 -0.05 -0.1 -0.25 -0.33 -0.5 -1)

# h_schedule
h_schedule=("True" "False")

# epochs
epochs=(20 30 50)

# learning rate
lr=(0.01 0.001 0.0001)

# lr_schedule
lr_schedule=("True" "False")

# sim dataset names
parent_directory="../../data/simulation/pyepo_shortest_path_reproduce"
# Read directory names into an array
IFS=$'\n' read -d '' -r -a directories < <(find "$parent_directory" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

#DSL
for dir in "${directories[@]}"; do 
    for sch in "${fin_diff_sch[@]}"; do
        for h in "${h_exp[@]}"; do
            for h_sch in "${h_schedule[@]}"; do
                for e in "${epochs[@]}"; do
                    for l in "${lr[@]}"; do
                        for lr_s in "${lr_schedule[@]}"; do
                            run_name="DSL_fin-diff-sch-${sch}_h-${h}_h-sch-${h_sch}_epoch-${e}_lr-${l}_lr-sch-${lr_s}"
                            command="python3 ../../pyepo_shortest_path_reproduce.py dataset.dataset_path='${dir}' loss_func.finite_diff_sch=$sch loss_func.h_exp=$h loss_func.h_sch=$h_sch model.epochs=$e model.lr=$l model.lr_schedule=$lr_s" 
                            echo $command
                            
                            sbatch -J "$run_name" slurm_shortest_path_exp.sh "$command"  
                        done
                        sleep 1
                    done
                done
            done
        done
    done
done


# SPO
for dir in "${directories[@]}"; do 
    for e in "${epochs[@]}"; do
        for l in "${lr[@]}"; do
            for lr_s in "${lr_schedule[@]}"; do
                run_name="SPO_epoch-${e}_lr-${l}_lr-sch-${lr_s}"
                command="python3 ../../pyepo_shortest_path_reproduce.py dataset.dataset_path='${dir}' loss_func.loss_name=SPO loss_func.loss_func._target_=pyepo.func.spoplus.SPOPlus model.epochs=$e model.lr=$l model.lr_schedule=$lr_s" 
                echo $command
                sbatch -J "$run_name" slurm_shortest_path_exp.sh "$command" 
            done
            sleep 1
        done
    done
done