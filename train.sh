#!/usr/bin/env bash

srun --gres=gpu:1,gmem:10GB --pty --mem=10G python -W ignore train.py
