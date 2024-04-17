# jatchat
Jatmo inspired universal LLM chat prompt injection defender

Requirements:
make sure you have cuda (and possibly cudnn) drivers
make sure you have torch and torchvision https://pytorch.org/get-started/locally/
windows users also need to enable the running application to have gpu access

jatChat3.py has the latest working version using [octopus-v2]([url](https://huggingface.co/NexaAIDev/Octopus-v2))

Necessary Variable Changes:
other models should work as well if you change the MODEL_PATH variable to another directory 
current settings use a BATCH_SIZE of 6, if gpu vram runs out lower BATCH_SIZE

