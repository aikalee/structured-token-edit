## Task
The model is aimed to fixed the outputs of a upstream model, and the outputs are the dependency relations in tree format.
## Data
### Input Format
The data is tranformed from linearized tree format to custom data format call Structured Token, the token (referring to POS here, since each POS is representing a token) itself, left and right brackets that wrap around the token are feeded into the model as three features:

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
Due to the complexity of the task, I decomposed the task into two parts and built two independent models to handle the two tasks. The first model called StructureTokenGate predicts where the corrections should happen, and the other model called StructuredTokenDeocder is responsible for predicting the correct left and right labels for each structured token.
