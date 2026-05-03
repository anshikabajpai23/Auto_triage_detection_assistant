Base model: "meta-llama/Llama-3.2-1B"
SFT model: "meta-llama/Llama-3.2-1B"
Dataset size: 154,795
Train size: 123836
Validation size: 15,479 
Test size: 15,480
Task type: Classification of Error logs
Input format: "Critical bug found in system32 file"
Expected output format: severity: P0| Team: Infra
Training method: full fine-tune / LoRA / QLoRA
Epochs: 3
Learning rate: 2.0e-4
Batch size: ???
Max sequence length: 512
Train loss: 1.5025
Eval loss: 1.4326
Eval metric: Macro-F1, accuracy, confusion matrix per class, failure rate



Is input clear?
- yes
Is expected answer correct?
- npt sure, but seems right
Is output format consistent?
 - yes
Are labels balanced?
 - no, P3 dominates
Are there duplicate examples?
 - no
Are train/test examples leaking?
 - no
Are answers too templated?
 - no, standard formatting


 Class Imbalance?
  - Yes, severe