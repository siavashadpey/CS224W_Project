# Hyperparameter Tuning Results

- Job Name: cs224w-hptuning-20251130-204852

## Hyperparameter Tuning Results
Total trials: 22, Successful trials: 21, Failed trials: 0, Stopped trials: 0

### Top 5 Trials
| trial_id | state | batch_size | hidden_dim | learning_rate | masking_ratio | num_decoder_layers | num_encoder_layers | val_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
17 | SUCCEEDED | 16.0 | 256.0 | 0.0001 | 0.300000 | 4.0 | 4.0 | 2.483737
16 | SUCCEEDED | 16.0 | 256.0 | 0.0001 | 0.313994 | 4.0 | 4.0 | 2.648540
19 | SUCCEEDED | 16.0 | 256.0 | 0.0001 | 0.300000 | 3.0 | 4.0 | 3.077179
15 | SUCCEEDED | 16.0 | 128.0 | 0.0001 | 0.412529 | 4.0 | 4.0 | 3.163890
20 | SUCCEEDED | 16.0 | 256.0 | 0.0001 | 0.300029 | 3.0 | 4.0 | 3.272481

------
### Parameter Correlations with val_loss
- learning_rate            : -0.445
- hidden_dim               : -0.565
- num_encoder_layers       : -0.274
- num_decoder_layers       :  0.255
- masking_ratio            :  0.488
- batch_size               :  0.802

----
### Best Configuration
- Trial ID: 17
- Val Loss: 2.483737

------------

### Hyperparameters:
- learning_rate            : 0.0001
- hidden_dim               : 256.0
- num_encoder_layers       : 4.0
- num_decoder_layers       : 4.0
- masking_ratio            : 0.3
- batch_size               : 16.0


## Key Insights
1. Learning rate: Top 5 trials had learning rate = 0.0001
2. Model size: Top 4/5 trials had hidden_dim=256 worked best
3. Depth: Top 5 trials had num_encoder_layers=4, and 3/5 top trials had num_decoder_layers=4
4. Masking: Top 4/5 trials had masking_ratio=0.3, but this could be because task is less challenging with lower masking rate
5. Batch: Top 8 trials had batch=16

## Next Steps
1. Perform full training of 1000 epochs for trials 17, 14, 15 which have a range of masking ratios to see if increased masking ratio improves downstream task