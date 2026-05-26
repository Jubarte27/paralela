#!/bin/bash
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

python3 -m venv "$HERE/.venv"

source "$HERE/.venv/bin/activate"

pip install --upgrade pip
pip install tensorflow-cpu scipy


# sudo sysctl -w kernel.yama.ptrace_scope=0