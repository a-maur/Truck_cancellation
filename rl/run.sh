WITH_SUMMARY=0 bash run_ppo.sh \
  --early-stop-enabled \
  --early-stop-warmup 100 \
  --early-stop-window 40 \
  --early-stop-check-every 10 \
  --early-stop-patience 3 \
  --early-stop-actor-slope-threshold 1e-4 \
  --early-stop-critic-slope-threshold 5e-4

