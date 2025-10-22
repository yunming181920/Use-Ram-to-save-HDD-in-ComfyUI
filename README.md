### Usage Instructions

Locate the `load_torch_file` function in `ComfyUI/Comfy/utils.py`, and replace it with the function provided in this project.  
Make sure to also copy the necessary import statements.

> ⚠️ **Note:**  
> This approach trades memory for speed, so it requires a large amount of RAM.  
> It is recommended to use at least **64 GB of memory**, or enable **virtual memory on an SSD** if sufficient RAM is not available.

### Performance Improvement

Previously, loading a large model took over **20 minutes**,  
but now it only takes **1 minute and 50 seconds**.  
Although it's still slower than loading from an SSD,  
the model switching time is now **much more acceptable**.

