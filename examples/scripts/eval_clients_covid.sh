#!/bin/bash
# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl_uchicago.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid.yaml \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_eqweight_covid_covid19newsplit1_anl_uchicago \
#     --load-model-filename checkpoint_28

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl_uchicago.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid.yaml \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_covid19newsplit1_anl \
#     --load-model-filename checkpoint_30
    
    
# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl_uchicago.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid.yaml \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_covid19newsplit1_anl_uchicago \
#     --load-model-filename checkpoint_30

# python funcx_sync.py \
#     --client_config configs/clients/covid19newsplit2_anl_uchicago.yaml \
#     --config configs/fed_avg/funcx_fedavg_covid.yaml \
#     --clients-test \
#     --load-model \
#     --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_covid19_uchicago \
#     --load-model-filename checkpoint_30

python funcx_sync.py \
    --client_config configs/clients/covid19newsplit2_anl_uchicago_data_norm.yaml \
    --config configs/fed_avg/funcx_fedavg_covid.yaml \
    --clients-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/outputs_CovidDataset_ServerFedAvg_Adam_funcx_fedavg_covid_covid19newsplit2_anl_data_norm \
    --load-model-filename checkpoint_30