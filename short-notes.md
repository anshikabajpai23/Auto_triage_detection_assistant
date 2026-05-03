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

Eval has blanks for both team and severity prediction.
Team - 5000/15500
Severity - 2500/15500
severity:P3 | team:unknown-platforms | team:unknown-platforms | team:unknown-platform
severity:P2 | team:unknown | issue:unreachable | steps to reproduce: [none]
In severity we are getting junk values in the output, where the model is not even predicting what kind of severity it is. It just repeats fragments of input. Why????

Assumptions:
P0 to P3/P4 - not enough info in title/body
Filed by - P0 to P4 - They are generally P4, that's why model classified it as P4 even though it was P0

** We can use other dataset as eval and using as out of domain eval, maybe rlhf**


check prompt mismatch?
 - No

Step 5: Compare base vs SFT on same examples?
format error
wrong class
hallucination
too generic
missed instruction
partial answer
refusal issue
repetition

Step 6: Check if SFT only learned format?
Is SFT output more structured than base?
 - Yes
Is the actual answer better?
Does it copy training style?
 - Yes
Does it still miss reasoning?



Train loss high, eval loss high
→ underfitting

Train loss low, eval loss high
→ overfitting

Train loss decreasing smoothly
→ training probably ran

Loss weird / NaN / flat
→ training bug

Run SFT inference on training as well. Same number of samples as eval

Reevaluate base VS SFT
- Shuld we add few shot prompting or optimize the prompt

What other metrics should we see?

