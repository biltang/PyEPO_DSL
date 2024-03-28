#!/bin/bash
mkdir -p ./outputs/slurmlogs

# finite diff scheme
fin_diff_sch=("B" "C" "F")

# h_exp 
h_exp=(-0.01 -0.1 -0.25 -0.5 -1) #-0.05 

# h_schedule
h_schedule=("True" "False")

# epochs
epochs=(50)

# learning rate
lr=(0.01 0.001) # 0.0001)

# lr_schedule
lr_schedule=("True" "False")

# MSE Init
MSE_init=("True" "False")

# sim dataset names
parent_directory="../../data/simulation/pyepo_shortest_path_reproduce"
# Read directory names into an array
IFS=$'\n' read -d '' -r -a directories < <(find "$parent_directory" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)


# #DSL
for dir in "${directories[@]}"; do
    if [[ $dir != n_100_* ]]; then
        continue # Skip this directory
    fi

    for sch in "${fin_diff_sch[@]}"; do
        for h in "${h_exp[@]}"; do
            for h_sch in "${h_schedule[@]}"; do
                for e in "${epochs[@]}"; do
                    for l in "${lr[@]}"; do
                        for lr_s in "${lr_schedule[@]}"; do
                            for m_init in "${MSE_init[@]}"; do
                                run_name="DSL_fin-diff-sch-${sch}_h-${h}_h-sch-${h_sch}_epoch-${e}_lr-${l}_lr-sch-${lr_s}_MSE-init-${m_init}"
                                command="python3 ../../pyepo_shortest_path_reproduce.py dataset.dataset_path='${dir}' loss_func=dsl loss_func.finite_diff_sch=$sch loss_func.h_exp=$h loss_func.h_sch=$h_sch model.epochs=$e model.lr=$l model.lr_schedule=$lr_s model.mse_init=$m_init" 
                                echo $command
                                sbatch -J "$run_name" slurm_shortest_path_PnO_exp.sh "$command" 

                            done
                        done
                        
                    done
                done
            done
        done

        sleep 5

    done
done


# SPO
# for dir in "${directories[@]}"; do
#     if [[ $dir != n_100_* ]]; then
#         continue # Skip this directory
#     fi

#     for e in "${epochs[@]}"; do
#         for l in "${lr[@]}"; do
#             for lr_s in "${lr_schedule[@]}"; do

#                 for m_init in "${MSE_init[@]}"; do
#                     run_name="SPO_epoch-${e}_lr-${l}_lr-sch-${lr_s}_MSE-init-${m_init}"
#                     command="python3 ../../pyepo_shortest_path_reproduce.py dataset.dataset_path='${dir}' loss_func=spo model.epochs=$e model.lr=$l model.lr_schedule=$lr_s model.mse_init=$m_init" 
#                     echo $command
#                     sbatch -J "$run_name" slurm_shortest_path_PnO_exp.sh "$command" 
#                 done
#             done
#             sleep 1
#         done
#     done
# done