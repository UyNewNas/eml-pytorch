<!--
 * @Author: slava
 * @Date: 2026-04-21 16:01:57
 * @LastEditTime: 2026-04-22 08:41:33
 * @LastEditors: ch4nslava@gmail.com
 * @Description: 
 * 
-->
# eml-pytorch
Hardware-efficient Exp-Minus-Log (EML) operator for PyTorch. A unified activation primitive for neuro-symbolic AI and edge deployment.


## Quick Start

### Install
```bash
pip install git+https://github.com/UyNewNas/eml-pytorch.git
```

### Single Node Example
```python
from eml_pytorch import EMLNode
...
```
Training output:
```
Epoch   0, Loss: 5.311534
...
最终损失: 0.051967
```

### Two-Node Network Example
```python
from eml_pytorch import TinyEMLNet
...
```
Training output:
```
Epoch   0, Loss: 6.100719
...
最终损失: 0.176003
```
```

> *Note: On Windows, Triton acceleration is not available due to lack of official support. For GPU performance testing, please use WSL2 or native Linux.*