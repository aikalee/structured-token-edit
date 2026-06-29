## Data
### Input Format
The data is tranformed from linearized tree format to custom structured token (overlap=3), the token (referring to POS here, since each POS is representing a token) itself, left and right brackets that wrap around the token are feeded into the model as three features:

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
