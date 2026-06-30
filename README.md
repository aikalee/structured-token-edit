## Task
The model is aimed to fixed the outputs of a upstream model, and the outputs are the dependency relations in tree format.
## Data
### Input Format
The data is tranformed from linearized tree format to custom data format call Structured Token. In linearized tree format, the token are delexicalized (i.e., represented by their corresponding POS). The token itself, left and right brackets that wrap around the token are feeded into the model as three features:

In the `base` field, `left` includes to all the left brackets to the token, and `right` containa all the right brackets to the token before seeing the next left brackets.

After a series of experiments, I found that including only left or right brackets is not differentiable enough for the model, and therefore added a field `overlap`. In the `overlap` field, the `left` of the previous token will overlap will the `right` of the next token. The max overlapping length is 3.
```
(TOP (ROOT (ADV (NMOD (NMOD NOUN )NMOD NOUN )NMOD NOUN (CJTN CONJ (CJT (NMOD NOUN )NMOD NOUN )CJT )CJTN )ADV VERB )ROOT )TOP
```
```
{
  "token": "NOUN",
  "source":
    {
      "base":
        {
          "left": ["(TOP", "(ROOT", "(ADV", "(NMOD", "(NMOD"],
          "right": [")NMOD"],
        },
      "overlap":
        {
          "left": ["(TOP", "(ROOT", "(ADV", "(NMOD", "(NMOD", ")NMOD"],
          "right": ["(ADV", "(NMOD", "(NMOD", ")NMOD", "NOUN", ")NMOD"],
        },
    },

```

### Output format
The output format of the model is partly the same with the input format, only the fields `token`, `left` and `right` remains in the output of the model:
```
{
  "token": "NOUN",
  "left": ["(TOP", "(ROOT", "(ADV", "(NMOD", "(NMOD"],
  "right": [")NMOD"],
},

```

## Model Architecture
Originally, I and my instructor planned to use T5 for this downstream fixing task, and both inputs and outputs are in linearized tree strucutre. However, I discovered two major problems. First, there are a lot of keeping and a little bit of editing in a downstream fixing task. The editing samples have weak signals. Since the decoder inputs and the hidden states are mostly co-occuring, the sentence-level decoder learns to copy the hidden states and leads to serious copy bias. Second, as autoregression is less constrainted, the model does not guarantee to output all the anchor token (POS) and sometimes cut the sentence off. I used token-level decoder to ensure every anchor token is in the outputs and ensure the basic structural validity.

Due to the complexity of the task, I decomposed the task into two parts and built two independent models to handle the two tasks. The first model called StructureTokenGate predicts where the corrections should happen, and the other model called StructuredTokenDeocder is responsible for predicting the correct left and right labels for each structured token. The decoder is limited to token level instead of sentence level.

The StructuredTokenGate return 1 and 0; 1 means edit needed and 0 refers to keep.

Since there are two independent models, there is a predictor that combines the results given by the two models. If the StructuredTokenGate returns 1, the predictions from the StructureTokenGate will be adopted. In the another way round, if the gate returns 0, the original predictions from the upstream model will be kept.

## Results
### Decoder

