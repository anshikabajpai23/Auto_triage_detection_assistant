import wandb
run = wandb.init(project='incident-triage', name='test-run', mode='online')
wandb.log({'test_metric': 42})
run.finish()
print('W&B working!')
