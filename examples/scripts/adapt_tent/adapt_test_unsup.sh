#!/bin/bash
FL_FOLD="outputs_funcx_fedavg_covid_sgd_lr0.003_covid19_uchicago_v2_midrc_v2"
FL_CKPT="best"

# Dataset selection
case $1 in
  cohen)
    CLIENT_CFG="covid19newsplit3_anl.yaml"
    ;;

  midrc)
    CLIENT_CFG="covid19_midrc_v2.yaml"
  ;;

  uchicago)
    CLIENT_CFG="covid19_uchicago_v2.yaml"
    ;;

  *)
    echo -n "unknown"
    ;;
esac

# Adaptation/Testing configs  
case $2 in
  none)
    ADAPT_CFG="adapt_test_unsup_none.yaml"
    ;;

  adapt)
    ADAPT_CFG="${1}_adapt_test_unsup_adapt.yaml"
  ;;

  fixed)
    ADAPT_CFG="adapt_test_unsup_fixed.yaml"
    ;;

  *)
    echo -n "unknown"
    ;;
esac

echo $ADAPT_CFG
python funcx_sync.py \
    --client_config configs/clients/$CLIENT_CFG \
    --config configs/tent/$ADAPT_CFG \
    --clients-adapt-test \
    --load-model \
    --load-model-dirname log_funcx_appfl/server/$FL_FOLD \
    --load-model-filename $FL_CKPT