#!/bin/bash

readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

name_from=$1; shift
name_to=$1; shift
dir_from=${1:-$_DIR}; shift
dir_to=${1:-$_DIR}; shift
mv_cp=${1:-mv}; shift
force_delete=${1:-false}; shift

readonly DATADIR="data/vhdl"
readonly CONFIGDIR="config"
readonly MODELDIR="models"
readonly LOGDIR="logs"
readonly TESTDIR="tests"


# Check if source save dir exist
if [ ! -d "${dir_from}/${MODELDIR}/${name_from}" ] || [ ! -d "${dir_from}/${LOGDIR}/${name_from}" ] || [ ! -d "${dir_from}/${TESTDIR}/${name_from}" ] || [ ! -d "${dir_from}/${DATADIR}/${name_from}" ] || [ ! -d "${dir_from}/${CONFIGDIR}/${name_from}" ]; then
        echo "Save directories for ${name_from} doesn't exist at ${dir_from}" 1>&2
        exit 125
fi


# Check if target save dir already exist
if [ -d "${dir_to}/${MODELDIR}/${name_to}" ] || [ -d "${dir_to}/${LOGDIR}/${name_to}" ] || [ -d "${dir_to}/${TESTDIR}/${name_to}" ] || [ -d "${dir_to}/${DATADIR}/${name_to}" ] || [ -d "${dir_to}/${CONFIGDIR}/${name_to}" ]; then
        if [ ${force_delete} == "true" ]; then
                rm -rf ${dir_to}/${MODELDIR}/${name_to}
                rm -rf ${dir_to}/${LOGDIR}/${name_to}
                rm -rf ${dir_to}/${TESTDIR}/${name_to}
                rm -rf ${dir_to}/${DATADIR}/${name_to}
                rm -rf ${dir_to}/${CONFIGDIR}/${name_to}
        else
                echo "Save directories for ${name_to} already exists at ${dir_to}.  Execute with 6th argument 'true' to remove them" 1>&2
                exit 125
        fi
fi


# Move / Copy the files
if [ ${mv_cp} == "cp" ]; then
        mv_cp="cp -r"
fi
mkdir -p ${dir_to}/${MODELDIR}
mkdir -p ${dir_to}/${LOGDIR}
mkdir -p ${dir_to}/${TESTDIR}
mkdir -p ${dir_to}/${DATADIR}
mkdir -p ${dir_to}/${CONFIGDIR}
${mv_cp} ${dir_from}/${MODELDIR}/${name_from} ${dir_to}/${MODELDIR}/${name_to}
${mv_cp} ${dir_from}/${LOGDIR}/${name_from} ${dir_to}/${LOGDIR}/${name_to}
${mv_cp} ${dir_from}/${TESTDIR}/${name_from} ${dir_to}/${TESTDIR}/${name_to}
${mv_cp} ${dir_from}/${DATADIR}/${name_from} ${dir_to}/${DATADIR}/${name_to}
${mv_cp} ${dir_from}/${CONFIGDIR}/${name_from} ${dir_to}/${CONFIGDIR}/${name_to}
