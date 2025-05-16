echo Activating environment
source /research/milsrg1/user_workspace/ab2810/miniconda3/bin/activate
cd /research/milsrg1/user_workspace/ab2810/repos/diffusion-separation
conda activate wv_env_env
echo Python version used:
python -V
echo Starting training...
python /research/milsrg1/user_workspace/ab2810/IIB_Proj/diffusion-separation/train.py
echo ...training finished
