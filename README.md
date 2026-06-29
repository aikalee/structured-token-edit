## Data
### Input Format
The data is tranformed from linearized tree format to custom structured token (overlap=3), the token (referring to POS here, since each POS is representing a token) itself, left and right brackets that wrap around the token are feeded into the model as three features:
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
