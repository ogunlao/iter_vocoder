# Itererative Vocoders

This repository implements the iterative vocoders as described in the paper,
"Beyond Griffin-Lim: Improved Iterative Phase Retrieval for Speech by Tal Peer et al"

[Paper link on arxiv](https://arxiv.org/abs/2205.05496)

Note that the vocoder implementations are for academic purposes and they are not optimized in terms of speed.

## The iterative algorithms implemented

- Griffin-Lim Algorithm (GLA)
- Fast Griffin-Lim (FGLA)
- Relaxed Averaged Alternating Reflections (RAAR)
- Difference Map (DiffMap)
- Alternating Direction Method of Multipliers (ADMM)
- Hybrid algorithms (where any of the above algorithms can be combined)

## How to use

Clone the repository

```bash
git clone github.com/iter_vocoder
```

Install requirements from `requirements.txt` i.e. librosa and numpy

```bash
cd iter_vocoder
pip install -r requirements.txt
```
Then, run the following code in a jupyter notebook or a python script

```python
import librosa
from iter_vocoder import (GriffinLim, FastGriffinLim, RAAR, 
                          DiffMap, ADMM, HybridVocoder)

# stft parameters
hop_length=256
win_length=1024
n_fft=1024
window="hann"
center=True

sampling_rate=16000

# load the spectrogram e.g extract spectrogram from an audio
audio, sr = librosa.load("sample_audio.wav", 
                         sr=sampling_rate)
complex_spec = librosa.stft(y=audio, 
                            hop_length=hop_length, 
                            win_length=win_length,
                            n_fft=n_fft,
                            window=window,
                            center=center,)

magnitude, phase = librosa.magphase(complex_spec)

# To use griffin-lim

# a. Initialize griffin-lim
vocoder = GriffinLim(n_iter=20,
                     hop_length=hop_length, 
                     win_length=win_length,
                     n_fft=n_fft,
                     window=window,
                     center=center,)
# b. use the vocode method 
gen_audio = vocoder.vocode(magnitude)

# c. To give an initial phase to vocoder
gen_audio = vocoder.vocode(magnitude, init_phase=phase)

# To use one or more iterative vocoders together aka hybrid vocoders

# parameters pertaining to each vocoder
# i.e first apply fast griffin-lim for 60 iterations, then raar for the last 40 iterations, for a total of 100 iterations
param_dict = {"fgla": {
                "n_iter": 60,
              },
              "raar": {
                "n_iter": 40,
              }
              }
# * You can choose among "gla", "fgla", "admm", "diffmap" and "raar"

# parameters to be applied to all vocoders e.g stft parameters
stft_args = dict(
    hop_length=hop_length,
    win_length=win_length,
    window=window,
    center=center,
    n_fft=n_fft,
)

hybrid_voc = HybridVocoder(param_dict, stft_args)
gen_audio = vocoder_hybrid.vocode(magnitude)

# You can also give an initial phase
gen_audio = vocoder.vocode(magnitude, init_phase=phase)
```

## Todo

- Add tests
- Compare implementation with results in paper

## Contributors

- Sewade Ogun
