# AITF-MCTS

The Monte Carlo tree search to generalize aitf behaviour cloning policy

### To install:

* Use conda if possible:
conda install --file requirements.txt

* Download the Trajair dataset to the dataset folder from: [Link](https://kilthub.cmu.edu/articles/dataset/TrajAir_A_General_Aviation_Trajectory_Dataset/14866251)

* Extract and make sure the file structure looks like this.
```
 dataset
│   └── 7days1
│       ├── processed_data
│       │   ├── test
```

### To run
* Use `sbatch_run.sh` to run in clusters.
* Use `python play.py --model_weights goalGAIL1b_60.pt` to debug.


### Maintainer
Jay Patrikar
jaypat@cmu.edu