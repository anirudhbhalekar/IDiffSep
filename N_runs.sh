echo Activating environment
source /research/milsrg1/user_workspace/ab2810/miniconda3/bin/activate
cd /research/milsrg1/user_workspace/ab2810/IIB_Proj/diffusion-separation

conda activate wv_env_env
echo Python version used:
python -V

echo Starting evaluation...

i=30

echo "Running with $i steps"
python /research/milsrg1/user_workspace/ab2810/IIB_Proj/diffusion-separation/evaluate.py tensorboard_sessions/MOD_RUN_CONT/version_2/checkpoints/epoch-031_si_sdr-16.366.ckpt \
    -o /research/milsrg1/user_workspace/ab2810/IIB_Proj/diffusion-separation/results \

