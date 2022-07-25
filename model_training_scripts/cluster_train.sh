#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --job-name="llm-finetune"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate ms21ddr_llms

# srun python train_models.py with 'run_name="dn_t5_tiny_enc/eraser_esnli/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True'
# srun python train_models.py with 'run_name="dn_t5_mini_enc/eraser_esnli/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True'
# srun python train_models.py with 'run_name="dn_t5_small_enc/eraser_esnli/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True'
# srun python train_models.py with 'run_name="dn_t5_base_enc/eraser_esnli/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True'
# srun python train_models.py with 'run_name="dn_t5_tiny_enc/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002"
# srun python train_models.py with 'run_name="dn_t5_mini_enc/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002"
# srun python train_models.py with 'run_name="dn_t5_small_enc/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002"
# srun python train_models.py with 'run_name="dn_t5_base_enc/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002"
# srun python train_models.py with 'run_name="bert_base_uncased/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002"
# srun python train_models.py with 'run_name="gpt2_small/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002"
# srun python train_models.py with 'run_name="t5_base_enc/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002"
# srun python train_models.py with 'run_name="roberta_base/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002"
# srun python train_models.py with 'run_name="bert_base_uncased/eraser_esnli/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"'
# srun python train_models.py with 'run_name="gpt2_small/eraser_esnli/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"'
# srun python train_models.py with 'run_name="t5_base_enc/eraser_esnli/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"'
srun python ./model_training_scripts/train_models.py with 'run_name="roberta_base/eraser_esnli/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"'
# srun python train_models.py with 'run_name="gpt2_small/spurious_sst/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"'
# srun python train_models.py with 'run_name="gpt2_small/mnli/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"'
# srun python train_models.py with 'run_name="gpt2_small/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"'
# srun python train_models.py with 'run_name="gpt2_small/eraser_esnli/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"'

# srun python train_models.py with 'run_name="dn_t5_tiny_enc/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002" "use_early_stopping=False"
# srun python train_models.py with 'run_name="dn_t5_mini_enc/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002" "use_early_stopping=False"
# srun python train_models.py with 'run_name="dn_t5_small_enc/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002" "use_early_stopping=False"
# srun python train_models.py with 'run_name="dn_t5_base_enc/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002" "use_early_stopping=False"
# srun python train_models.py with 'run_name="bert_base_uncased/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002" "use_early_stopping=False"
# srun python train_models.py with 'run_name="gpt2_small/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002" "use_early_stopping=False"
# srun python train_models.py with 'run_name="t5_base_enc/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002" "use_early_stopping=False"
# srun python train_models.py with 'run_name="roberta_base/eraser_cose/cls-finetune"' 'data_cache_dir="/scratch-ssd/ms21ddr/data/hf_language_datasets"' 'disable_tqdm=True' 'output_dir="/scratch-ssd/ms21ddr/model_outputs"' "batch_size=32" "num_epochs=40" "lr=.00002" "use_early_stopping=False"