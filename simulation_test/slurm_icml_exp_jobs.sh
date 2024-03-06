#!/bin/bash
mkdir -p ./outputs/slurmlogs

# config changes

# training n
n=(20 40 60 80 100 120 140 160 180 200)

# data slope m
m=(-4 0)

# finite diff scheme
fin_diff_sch=("B" "C")

for i in "${n[@]}"; do
    for j in "${m[@]}"; do
        for k in "${fin_diff_sch[@]}"; do
            run_name="icml-exp-DSL_n_${i}_m_${j}_sch_${k}"
            echo "python3 ../dsl_icml_reproduce.py sim.n=$i sim.y_config.m=$j model.finite_diff_sch=$k"
            sleep 1
            sbatch -J "$run_name" slurm_icml_exp.sh "python3 ../dsl_icml_reproduce.py sim.n=$i sim.y_config.m=$j model.finite_diff_sch=$k"
        done
    done
done

# SPO and MSE runs
model_name=("SPO" "MSE")

for i in "${n[@]}"; do
    for j in "${m[@]}"; do
        for k in "${model_name[@]}"; do
            run_name="icml-exp-${k}_n_${i}_m_${j}"

            if [ "$k" = "SPO" ]; then
                command="python3 ../dsl_icml_reproduce.py model.model_name=$k sim.n=$i sim.y_config.m=$j model.model_func._target_=pyepo.func.spoplus.SPOPlus"
            else
                command="python3 ../dsl_icml_reproduce.py model.model_name=$k sim.n=$i sim.y_config.m=$j"
            fi

            echo $command
            sleep 1
            sbatch -J "$run_name" slurm_icml_exp.sh "$command"
        done
    done
done
   
    