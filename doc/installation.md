# Developer installation

## Prerequisites


### Linux/Unix

Install Arb (**version >= 2.20.0**):
```shell
sudo apt install libflint-arb-dev=1:2.21.1-2
# sudo apt install libflint-dev libgmp-dev
```

#### Troubleshooting

If there's no such version of Arb:

- Run `sudo apt update` first.
- If still no such version, add the following line to your `/etc/apt/sources.list` (or create a new file `/etc/apt/sources.list.d/sid.list` and add the following line to it):
  `deb http://ftp.debian.org/debian sid main`. Then run `sudo apt update` again.
- If there are errors like `The following signatures couldn't be verified because the public key is not available`,
  run `apt-key adv --keyserver keyring.debian.org --recv-keys [public key]` for each public key. Then run `sudo apt update` again.
  

### Mac OS

- Install arb  `sudo apt install arb`
- Download cmake
- Run `$ cmake cmake CMakeLists.txt`
  This will generate a `Makefile`
- Run `$ make`