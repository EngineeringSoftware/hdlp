#!/bin/bash

readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# readonly DATADIR="data/vhdl"
readonly CONFIGDIR="config"
readonly MODELDIR="models"
readonly LOGDIR="logs"
readonly TESTDIR="tests"

readonly BACKUP_NAME="backup-savedir-$(date +"%m%d%Y_%H%M%S")"

mkdir ${BACKUP_NAME}
mkdir ${BACKUP_NAME}/data
cp -r data/vhdl ${BACKUP_NAME}/data
cp -r ${CONFIGDIR} ${BACKUP_NAME}
cp -r ${MODELDIR} ${BACKUP_NAME}
cp -r ${LOGDIR} ${BACKUP_NAME}
cp -r ${TESTDIR} ${BACKUP_NAME}

tar czf ${BACKUP_NAME}.tgz ${BACKUP_NAME}/
