#!/bin/bash
mkdir -p ./outputs/slurmlogs

# epochs
epochs=(50)

# learning rate
lr=(0.01 0.001 0.0001)

# lr_schedule
lr_schedule=("True" "False")

# sim dataset names
parent_directory="../../data/simulation/pyepo_shortest_path_reproduce"
# Read directory names into an array
IFS=$'\n' read -d '' -r -a directories < <(find "$parent_directory" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

# MSE
for dir in "${directories[@]}"; do 
    # Check if the directory name starts with "n_100"
    if [[ $dir != n_100_* ]]; then #
        continue # Skip this directory
    fi

    for e in "${epochs[@]}"; do
        
        for l in "${lr[@]}"; do
            
            for lr_s in "${lr_schedule[@]}"; do
                

                run_name="MSE_epoch-${e}_lr-${l}_lr-sch-${lr_s}"
                command="python3 ../../pyepo_shortest_path_reproduce.py dataset.dataset_path='${dir}' loss_func=mse model.epochs=$e model.lr=$l model.lr_schedule=$lr_s" 
                echo $command
                            
                sbatch -J "$run_name" slurm_shortest_path_MSE_exp.sh "$command"  
            done
        done
        sleep 1
    done
done
