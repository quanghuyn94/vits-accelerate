# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

A customized version of VITS has been developed to accommodate emerging technologies.

- Please note that this is an experimental version, and there are still several issues that need to be addressed before it can be used. Some of the differences from the regular version include:

## Some of the differences from the regular version include:
- Use with PyTorch 2.
- Utilizes the Accelerate library.

## Use
Requires the following dependencies:
- Cmake.
- espeak.

Training:
1. First, you need to install Conda. I recommend using it to simplify the training process.
2. Create a Conda environment:
```bash 
conda create -n vits
```
3. Activate the environment and install the requirements:
```bash 
conda activate vits
pip install -r requirements.txt
```

4. Create a dataset for your own use:
- You can create a dataset similarly to how you create a dataset for the regular VITS version.
- Alternatively, you can create a custom dataset with the following structure:
```
<you dataset>
|__ train
|   |__ audio1.wav
|   |__ audio1.txt.cleaned
|   |__ audio2.wav
|   |__ audio2.txt.cleaned
|   |__ *.wav
|   |__ *.txt.cleaned
|   |__ <folder>
|   |   |__*.wav
|   |   |__*.txt.cleaned
|
|__ eval
|   |__ audio1.wav
|   |__ audio1.txt.cleaned
|   |__ audio2.wav
|   |__ audio2.txt.cleaned
|   |__ *.wav
|   |__ *.txt.cleaned
|   |__ <folder>
|   |   |__*.wav
|   |   |__*.txt.cleaned
```
- Alternatively, the structure could be like this:
```
<you dataset>
|__ audio1.wav
|__ audio1.txt.cleaned
|__ audio2.wav
|__ audio2.txt.cleaned
|__ *.wav
|__ *.txt.cleaned
|__ <folder>
|   |__*.wav
|   |__*.txt.cleaned
```
- If you're feeling lazy, you can use the preprocess script for a quicker solution. Use it as follows:
```bash
python preprocess.py --data_dir <you_dataset> --save_to <save to> --src_lang <language of audio> --text_cleaners <text_cleaners>
```
5. After you've finished all the setup hoopla, it's time to fire up the training frenzy with this command:
```bash
accelerate launch train_accelerate.py 
    -c <config> 
    -m <save to>
    --custom_dataset=<you dataset>
    --batch_size=16 
    --cache_spectrogram_to_disk
    --cache_spectrogram
```

Now, grab a bag of popcorn, kick your feet up, and watch the neural magic unfold. It's as simple as trying to teach your cat to breakdance.



## Issues:
- Currently, multi-speaker and so-vits training are not yet supported.
- Training speed remains unimproved, but it does save a bit of GPU memory.
- Bugs, errors, and code chaos are still in the mix - think of it as a digital adventure in the land of glitches.
- There might be a missing module or two.

Wishing you a whimsical and glitch-tolerant journey (❁´◡`❁)



