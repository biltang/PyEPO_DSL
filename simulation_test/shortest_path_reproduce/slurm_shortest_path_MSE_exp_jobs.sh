#!/bin/bash
mkdir -p ./outputs/slurmlogs

# epochs
epochs=(100)

# learning rate
lr=(0.01 0.001) # 0.1  0.0001)

# lr_schedule
lr_schedule=("True" "False")

# sim dataset names
parent_directory="../../data/simulation/pyepo_shortest_path_reproduce_wk_4_1_no_noise"
# Read directory names into an array
IFS=$'\n' read -d '' -r -a directories < <(find "$parent_directory" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

# MSE
# for dir in "${directories[@]}"; do 
#     # Check if the directory name starts with "n_100"
#     if [[ $dir != n_100_* ]]; then #
#         continue # Skip this directory
#     fi

#     for e in "${epochs[@]}"; do
        
#         for l in "${lr[@]}"; do
            
#             for lr_s in "${lr_schedule[@]}"; do
                

#                 run_name="MSE_epoch-${e}_lr-${l}_lr-sch-${lr_s}"
#                 command="python3 ../../pyepo_shortest_path_reproduce.py loss_func=mse dataset.dataset_path='${dir}' model.epochs=$e model.lr=$l model.lr_schedule=$lr_s" 
#                 echo $command
                            
#                 sbatch -J "$run_name" slurm_shortest_path_MSE_exp.sh "$command"  
#             done
#         done
#         sleep 1
#     done
# done

# Cosine
for dir in "${directories[@]}"; do 
    # Check if the directory name starts with "n_100"
    if [[ $dir != n_100_* ]]; then #
        continue # Skip this directory
    fi

    for e in "${epochs[@]}"; do
        
        for l in "${lr[@]}"; do
            
            for lr_s in "${lr_schedule[@]}"; do
                

                run_name="Cosine_epoch-${e}_lr-${l}_lr-sch-${lr_s}"
                command="python3 ../../pyepo_shortest_path_reproduce.py loss_func=cosine dataset.dataset_path='${dir}' model.epochs=$e model.lr=$l model.lr_schedule=$lr_s" 
                echo $command
                            
                sbatch -J "$run_name" slurm_shortest_path_MSE_exp.sh "$command"  
            done
        done
        sleep 1
    done
done

