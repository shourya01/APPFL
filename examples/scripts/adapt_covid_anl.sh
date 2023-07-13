#!/bin/bash
FL_FOLD="outputs_funcx_fedavg_covid_sgd_lr0.003_covid19_uchicago_v2_midrc"
FL_CKPT="best"
python funcx_sync.py \
    --client_config configs/clients/covid19newsplit2_anl.yaml \
    --config configs/fed_avg/funcx_fedavg_covid_sgd_fc_adapt_0.001.yaml \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/$FL_FOLD \
    --load-model-filename $FL_CKPT