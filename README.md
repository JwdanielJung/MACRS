## Usage

Run the script using the following command:

```
python main.py -d [DATASET_TYPE] -r [RECOMMEND_MULTIPLE]
```

### Arguments

- `-d`, `--datasettype`: Specify the dataset type. Choose between "MACRS" or "persona".
  - `MACRS`: Use the MACRS dataset
  - `persona`: Use our custom persona dataset

- `-r`, `--recommend_multiple`: Specify whether to use multiple recommendations. Use "True" or "False".
  - `True`: Enable multiple recommendations
  - `False`: Disable multiple recommendations

### Examples

1. Use MACRS dataset with multiple recommendations:
   ```
   python main.py -d MACRS -r True
   ```

2. Use persona dataset without multiple recommendations:
   ```
   python main.py -d persona -r False
   ```

3. Use default settings(MACRS with multiple recommendations):
   ```
   python main.py
   ```