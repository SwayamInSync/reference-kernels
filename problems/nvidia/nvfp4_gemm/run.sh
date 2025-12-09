rm -rf __pycache__ 
rm -rf ~/.cache/torch_extensions
mode=$1
POPCORN_FD=1 CUTE_DSL_ARCH=sm_100a python eval.py $mode task.yml