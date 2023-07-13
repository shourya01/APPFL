#!/bin/bash
FL_FOLD="outputs_funcx_fedavg_covid_sgd_lr0.003_covid19_uchicago_v2_midrc_v2"
FL_CKPT="best"

python funcx_debug.py \
    --client_config configs/clients/covid19newsplit2_anl.yaml \
    --config configs/fed_avg/funcx_fedavg_covid_adapt_tent.yaml \
    --clients-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/$FL_FOLD \
    --load-model-filename $FL_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_midrc_v2.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_adapt_tent.yaml \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$FL_FOLD \
#     --load-model-filename $FL_CKPT

# python funcx_sync.py \
#     --client_config configs/clients/covid19_uchicago_v2.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid_adapt_tent.yaml \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/$FL_FOLD \
#     --load-model-filename $FL_CKPT