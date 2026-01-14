#!/bin/bash
# This script submits the solution to the nvidia leaderboard using popcorn-cli.
#
# Available modes:
# - leaderboard: Submits to the official leaderboard.
# - profile: to get the ncu analysis for the kernel

mode=$1
popcorn-cli submit submission.py --gpu NVIDIA --leaderboard nvfp4_dual_gemm --mode $mode --no-tui
