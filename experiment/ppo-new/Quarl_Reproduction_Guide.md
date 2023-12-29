# Quarl Reproduction Guide

## Setup

### Docker (Recommended for Evaluation)

To use already setup environment, start the docker container via the following command.

```shell
docker run --name quarl --gpus all -itd co1lin/quarl
docker exec -it quarl zsh
cd quartz # /home/ubuntu/quartz
git checkout quarl-repro
```

### Manual Setup (Recommended for Development)

1. Setup the Python environment with [mamba](https://github.com/quantum-compiler/quartz/blob/master/INSTALL.md#install-from-source). 

2. Install Quartz following [this section](https://github.com/quantum-compiler/quartz/blob/master/INSTALL.md#install-from-source), in which a mamba (conda) environment `quartz` is created.

3. Install packages for Quarl:

    ```shell
    # at the root dir of quartz repo
    git checkout quarl-repro
    cd experiment/ppo-new
    mamba activate quartz
    mamba env update --file env_ppo.yml
    ```
    Check if the GPU-enabled PyTorch is installed. If not (only the CPU version is installed), uninstall `torch` via `pip` and install the GPU-enabled PyTorch following the instruction on its official website.

4. Log in to the wandb account: `wandb login`.

5. Download the checkpoints for AI models and create some folders:

    ```shell
    # at the root dir of quartz repo
    git checkout quarl-repro
    cd experiment/ppo-new # this is the folder for Quarl experiments
    
    wget https://share.la.co1in.me/ckpts.zip
    unzip ckpts.zip && rm ckpts.zip
    mkdir ftlog
    ```

## Reproduce

- Program entry: `ppo.py`
- Output:
    - local: `outputs`
    - online: wandb project webpages, e.g. https://wandb.ai/userXYZ/IBM-Finetune-6l-seed

Examples for Table 2, Nam Gate set:

```shell
# w/o rotation merging pre-processing
CKPT=ckpts/nam2_iter_384_6l.pt BS=4800 CIRC=mod_red_21 GPU=0 bash scripts/nam2_search.sh
CKPT=ckpts/nam2_iter_384_6l.pt BS=3200 CIRC=adder_8 GPU=1 bash scripts/nam2_search.sh

# w/ rotation merging pre-processing
CKPT=ckpts/nam2_rm_iter_404_6l.pt BS=4800 CIRC=mod_red_21 GPU=2 bash scripts/nam2_rm_search.sh
CKPT=ckpts/nam2_rm_iter_404_6l.pt BS=3200 CIRC=adder_8 GPU=3 bash scripts/nam2_rm_search.sh
```

Examples for Table 3, CNOT count on Nam Gate set:

```shell
# w/o rotation merging pre-processing
CKPT=ckpts/nam2_iter_111_6l_cx.pt BS=4800 CIRC=mod_red_21 GPU=0 bash scripts/nam2_search_cx.sh
CKPT=ckpts/nam2_iter_111_6l_cx.pt BS=3200 CIRC=adder_8 GPU=1 bash scripts/nam2_search_cx.sh
```

Examples for Table 4, IBM Gate set:

```shell
CKPT=ckpts/ibm_iter_921_6l.pt BS=4800 CIRC=barenco_tof_10 GPU=0 bash scripts/ibm_search_6l.sh
CKPT=ckpts/ibm_iter_921_6l.pt BS=2400 CIRC=adder_8 GPU=1 bash scripts/ibm_search_6l.sh
```

