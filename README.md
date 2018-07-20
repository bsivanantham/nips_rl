Fork of [original repository](https://github.com/fgvbrt/nips_rl) modified for NIPS 2018.

# Description
[Reason8.ai](https://reason8.ai/) code for 3th place [NIPS learning to run challenge](https://www.crowdai.org/challenges/nips-2017-learning-to-run).

We are porting this code to pytorch [here](https://github.com/Scitator/Run-Skeleton-Run)

# Installation:
### Automatic (old, might not work for new OpenSim version):
1) Get opensim package. You can use default package as described [here](https://github.com/stanfordnmbl/osim-rl) or
build by youself faster version [here](https://github.com/Scitator/opensim-core)
2) Run setup script:
    
        $ ./setup.sh

You may want to change conda env name in script and comment last line if not building opnesim by yourself 

3) If you experience theano errors try to move .theanorc file:

        $ mv ~/.theanorc ~/.theanorc.backup

### Manual (after having setup the OpenSim venv):
Install dependencies:

        $ conda install numpy scipy scikit-learn mkl theano
        $ pip install https://github.com/Lasagne/Lasagne/archive/master.zip

# Running code
1) Activate environment:

        $ source activate nips_rl_fast3

2) Run code with best flags:

        $ python run_experiment.py --accuracy 0.01 --modeldim 3D --prosthetic False --difficulty 0 --actor_layers \(64,64\) --critic_layers \(64,32\)

# Other
The final submitted model was trained in this [commit](https://github.com/fgvbrt/nips_rl/tree/e2ffeaa475c57c64bf6d4664b2ab47b46ecc1c6e/dpg3).

There are lot of branches with various ideas tested during competition but without documentation, you could check for example following branches:
    
   - [distributed ddpg with pyro4](https://github.com/fgvbrt/nips_rl/tree/farm/pyro_farm) inspired by [ctmarko repository](https://github.com/ctmakro/stanford-osrl)
   - [distributed CEM with pyro4](https://github.com/fgvbrt/nips_rl/tree/cem/pyro_farm) I am not sure that this is canonical implementation, it was done in the last night.
   - [we even tried to do planning](https://github.com/fgvbrt/nips_rl/tree/kr-uct/kr-uct) as described in [this article](https://www.ijcai.org/Proceedings/16/Papers/104.pdf) 

