# Reference notebook repairs
Source: `NEGSC(SourceCode)/NEGSC.ipynb`  Output: `reference/NEGSC_repaired.ipynb`

Only these changes exist. Anything else that differs is a port bug.

| Cell | ID | Change |
|---|---|---|
| 2 | C1 | pd.read_csv('../NF-BoT-IoT-v2.csv')  ->  pd.read_csv('data_raw/sample_3pct.csv') |
| 3 | C2 | groupby sample guarded off (cache already sampled) |
| 4 | R1 | data['label'] =  ->  data['label'] = le.fit_transform(data.label.values) |

## R1 is not optional
The released notebook contains a syntax error at the label-encoding cell:
`data['label'] = ` with no right-hand side. The file cannot be executed as
published. This is worth stating plainly in the write-up: the artifact that
accompanies the paper does not run.

## C1/C2 are configuration, not logic
They point the notebook at the same input the port consumes, which is what
makes the parity comparison meaningful. No modelling code is touched.
