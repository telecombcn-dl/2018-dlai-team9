#!/usr/bin/env bash

for i in "$@"
do
case $i in
    -p=*|--params=*)
    PARAMS="${i#*=}"
    shift # past argument=value
    ;;
    *)
            # unknown option
    ;;
esac
done

if [ "${PARAMS}" = "params_train"  ]
    then
    srun  --gres=gpu:1,gmem:11GB --pty  --mem=20G  python train.py -p"${PARAMS}"

fi
