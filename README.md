# Usage Instructions
### ImDisk Installation Guide

1. First, download **ImDisk** from [SourceForge](https://sourceforge.net/projects/imdisk-toolkit/).  
2. Install it using the default settings.

### Modify the code

Locate the `load_torch_file` function in `ComfyUI/Comfy/utils.py`, and replace it with the function provided in this project.  
Make sure to also copy the necessary import statements.

### Steps to start

1. First, run `imdisk.bat`.  
2. Then, launch **ComfyUI** or **Akie Launcher**.
3. Run your workflows.
4. Input 'q' to close the imdisk.bat and delete the temporary Z: drive
   
> ⚠️ **Note:**  
> This approach trades memory for speed, so it requires a large amount of RAM. It is recommended to use at least **64 GB of memory**, or enable **virtual memory on an SSD** if sufficient RAM is not available.
  
> Alternatively, you can change all references to the Z: drive in the code to a specific folder on your SATA SSD. This way, files will only be added or deleted within that folder.However, do not attempt this if you do not have coding experience.
 
> You can change the 27GB in .bat to the size of the largest model in your workflow.
### Performance Improvement

Previously, loading a large model took over **20 minutes**,  
but now it only takes **1 minute and 50 seconds**.  
Although it's still slower than loading from an SSD,  
the model switching time is now **much more acceptable**.

