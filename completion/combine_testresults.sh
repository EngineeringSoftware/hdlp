#!/bin/bash

if [ "${1}" != "--suffix" ]; then
        echo "Help: ${0} --suffix VAL"
        exit 1
fi
shift
suffix="${1}"; shift

#TODO: add validation set

readonly DST="../slpproject/_results/agn-pred/lpa-model-test${suffix}.txt"
readonly SRCDIR="../slpproject/_results/test-results"

function extract() {
        local model="${1}"; shift
        local suffix="${1}"; shift

        local f="${SRCDIR}/${model}${suffix}.json"
        if [ -e ${f} ]; then
                # TODO: round numbers
                bleu=$(grep '"bleu-AVG"' ${f} | cut -f2 -d':' | sed 's/.*0\.\(.\)\(.\).*/0\.\1\2/g')
                acc=$(grep '"acc-AVG"' ${f} | cut -f2 -d':' | sed 's/.*0\.\(.\)\(.\).*/0\.\1\2/g')
                echo "${model}" "${acc}" "${bleu}" "0.0" "0.0" >> ${DST}
        else
                echo "WARNING: missing ${f}"
        fi
}

MODELS=(Baseline
        LM-10gram
        LM-10gram+PA1
        LM-10gram+PA1-5
        LM-RNN
        LM-RNN+PA1
        LM-RNN+PA1-5
        S2S-LHS
        S2S-LHS+PA1
        S2S-LHS+PA1+Type
        S2S-LHS+PA2+Type
        S2S-LHS+PA3+Type
        S2S-LHS+PA4+Type
        S2S-LHS+PA5+Type
        S2S-LHS+PA1-2+Type
        S2S-LHS+PA1-3+Type
        S2S-LHS+PA1-4+Type
        S2S-LHS+PA1-5+Type
        S2S-LHS+PAEnsemb-1-5+Type
        S2S-LHS+PAConcat-1-5+Type
        S2S-LHS+PA1-5+Type-NoAttn
       )

rm -f ${DST}
for m in ${MODELS[@]}; do
        extract ${m} ${suffix}
done
