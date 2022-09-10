# Developer installation

## Code format guide

Please setup `pre-commit` following [pre-commit official page](https://pre-commit.com) first.

Usually, this can be done in your working Python environment by:

```shell
pip install pre-commit clang-format
# cd quartz-repo
pre-commit install
```

Then `pre-commit` will run automatically on `git commit` .

You can also manually run it by `pre-commit run --all-files` .

## Prerequisites

### Linux/Unix

Install Arb (**version >= 2.20.0**):
```shell
sudo apt install libflint-arb-dev=1:2.22.1-1
# sudo apt install libflint-dev libgmp-dev
```

#### Troubleshooting

If there's no such version of Arb:

- Run `sudo apt update` first.
- If still no such version, add the following line to your `/etc/apt/sources.list` (or create a new file `/etc/apt/sources.list.d/sid.list` and add the following line to it):
  `deb http://ftp.debian.org/debian sid main`. Then run `sudo apt update` again.
- If there are errors like `The following signatures couldn't be verified because the public key is not available`,
  run `apt-key adv --keyserver keyserver.ubuntu.com --recv-keys [public key]` for each public key. Then run `sudo apt update` again.


## Python virtual environment


Create a virtual environment
`python3 -m venv venv`

Activate
`source venv/bin/activate`

Install requirements.txt
`pip install -r requirements.txt`
