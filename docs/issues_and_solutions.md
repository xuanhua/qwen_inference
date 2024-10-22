
The common issues and solutions

1. **cannot import name 'split_torch_state_dict_into_shards' from 'huggingface_hub'**

```bash
pip install huggingface-hub==0.23.0
```
reference: https://stackoverflow.com/questions/78920095/cannot-import-name-split-torch-state-dict-into-shards-from-huggingface-hub

2. **Cannot display Chinese characters correctly in terminal**

```bash
export LANG=zh_CN.utf8
```