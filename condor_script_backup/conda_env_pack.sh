#!/bin/bash

echo "Preparing in-silico-hVOS.tar.gz in staging with current environment state"
conda deactivate
conda pack -n in-silico-hVOS --dest-prefix='$ENVDIR'
chmod 644 in-silico-hVOS.tar.gz
ls -sh in-silico-hVOS.tar.gz

cp in-silico-hVOS.tar.gz /staging/jjudge3/in-silico-hVOS-env.tar.gz
