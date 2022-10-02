from abc import ABC, abstractmethod

import librosa
import numpy as np


class IterVocoder:
    def __init__(
        self,
        n_iter=100,
        hop_length=None,
        win_length=None,
        n_fft=None,
        window="hann",
        center=True,
        dtype=None,
        length=None,
        pad_mode="constant",
        random_state=None,
    ):

        self.n_iter = n_iter
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.random_state = random_state
        self.dtype = dtype
        self.n_fft = n_fft
        self.win_length = win_length
        self.length = length
        self.pad_mode = pad_mode
        self.random_state = random_state

    @abstractmethod
    def vocode(self, stft, init_phase=None, return_phase=False):
        """Placeholder for the vocoder function.
        Implement the vocode function for different iterative vocoders.
        """

        raise NotImplementedError

    def project_onto_magspec(self, magspec, stft):
        r"""This finds the projection of spectrogram 'complex_spec' on the set of all complex
        spectrogram that has a known magnitude 'magspec'"""
        return magspec * stft / (np.abs(stft) + self.eps)

    def project_complex_spec(self, stft):
        """stft: Complex spectrogram"""

        # Invert with our current estimate of the phases
        inverse_stft = librosa.istft(
            stft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            dtype=self.dtype,
            length=self.length,
        )

        # Rebuild the spectrogram
        next_stft = librosa.stft(
            inverse_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
        )

        return next_stft

    def init_random_state(self, random_state):
        # To preserve the reproducibility of the iterative algorithm
        self.random_state = self.random_state or random_state
        if self.random_state is None:
            self.rng = np.random
        elif isinstance(random_state, int):
            self.rng = np.random.RandomState(seed=random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state


class GriffinLim(IterVocoder):
    def __init__(
        self,
        n_iter=100,
        hop_length=None,
        win_length=None,
        n_fft=None,
        window="hann",
        center=True,
        dtype=None,
        length=None,
        pad_mode="constant",
        random_state=None,
    ):
        super().__init__(
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window=window,
            center=center,
            dtype=dtype,
            length=length,
            pad_mode=pad_mode,
            random_state=random_state,
        )

        self.init_random_state(random_state)

    def vocode(self, magspec, init_phase=None, return_phase=False):
        phase = np.empty(magspec.shape, dtype=np.complex64)
        self.eps = librosa.util.tiny(phase)

        if init_phase is None:
            # randomly initialize the phase
            phase[:] = np.exp(2j * np.pi * self.rng.rand(*magspec.shape))
        else:
            phase[:] = init_phase / (np.abs(init_phase) + self.eps)

        # Initialize the complex spectrogram
        stft = magspec * phase

        for _ in range(self.n_iter):
            stft[:] = self.project_onto_magspec(magspec, stft)
            stft[:] = self.project_complex_spec(stft)

        # Return the final phase estimates
        phase[:] = stft / (np.abs(stft) + self.eps)

        gen_audio = librosa.istft(
            magspec * phase,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            dtype=self.dtype,
            length=self.length,
        )

        if return_phase:
            return gen_audio, phase
        return gen_audio


class FastGriffinLim(GriffinLim):
    """Implements the fast griffin-lim algorithm with momentum"""

    def __init__(
        self,
        n_iter=100,
        momentum=0.99,
        hop_length=None,
        win_length=None,
        n_fft=None,
        window="hann",
        center=True,
        dtype=None,
        length=None,
        pad_mode="constant",
        random_state=None,
    ):
        super().__init__(
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window=window,
            center=center,
            dtype=dtype,
            length=length,
            pad_mode=pad_mode,
            random_state=random_state,
        )

        self.init_random_state(random_state)
        self.momentum = momentum

    def vocode(self, magspec, init_phase=None, return_phase=False):
        phase = np.empty(magspec.shape, dtype=np.complex64)
        self.eps = librosa.util.tiny(phase)

        if init_phase is None:
            # randomly initialize the phase
            phase[:] = np.exp(2j * np.pi * self.rng.rand(*magspec.shape))
        else:
            phase[:] = init_phase / (np.abs(init_phase) + self.eps)

        # Initialize the complex spectrogram
        next_stft = magspec * phase
        stft_acc = next_stft[:]

        for _ in range(self.n_iter):
            # Store the previous iterate
            prev_stft = next_stft.copy()

            next_stft[:] = self.project_complex_spec(self.project_onto_magspec(magspec, stft_acc))

            # Update our phase estimates
            stft_acc[:] = next_stft - (self.momentum / (1 + self.momentum)) * prev_stft

        # Return the final phase estimate
        phase[:] = stft_acc / (np.abs(stft_acc) + self.eps)

        gen_audio = librosa.istft(
            magspec * phase,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            dtype=self.dtype,
            length=self.length,
        )

        if return_phase:
            return gen_audio, phase
        return gen_audio


class RAAR(IterVocoder):
    """Relaxed Averaged Alternating Reflections"""

    def __init__(
        self,
        n_iter=100,
        beta=0.9,
        hop_length=None,
        win_length=None,
        n_fft=None,
        window="hann",
        center=True,
        dtype=None,
        length=None,
        pad_mode="constant",
        random_state=None,
    ):
        super().__init__(
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window=window,
            center=center,
            dtype=dtype,
            length=length,
            pad_mode=pad_mode,
            random_state=random_state,
        )

        self.init_random_state(random_state)
        self.beta = beta

    def reflect_on_melspec_on_plane(self, magspec, stft):
        return 2 * self.project_onto_magspec(magspec, stft) - stft

    def reflect_on_complex_set(self, stft):
        return 2 * self.project_complex_spec(stft) - stft

    def vocode(self, magspec, init_phase=None, return_phase=False):
        phase = np.empty(magspec.shape, dtype=np.complex64)
        self.eps = librosa.util.tiny(phase)

        if init_phase is None:
            # randomly initialize the phase
            phase[:] = np.exp(2j * np.pi * self.rng.rand(*magspec.shape))
        else:
            phase[:] = init_phase / (np.abs(init_phase) + self.eps)

        # Initialize the complex spectrogram
        stft = magspec * phase

        for _ in range(self.n_iter):
            temp_stft = self.reflect_on_complex_set(
                self.reflect_on_melspec_on_plane(magspec, stft)
            )
            stft[:] = 0.5 * self.beta * (stft + temp_stft) + (
                1 - self.beta
            ) * self.project_onto_magspec(magspec, stft)

        # Return the final phase estimates
        phase[:] = stft / (np.abs(stft) + self.eps)

        gen_audio = librosa.istft(
            magspec * phase,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            dtype=self.dtype,
            length=self.length,
        )

        if return_phase:
            return gen_audio, phase
        return gen_audio


class DiffMap(IterVocoder):
    """Difference Map Iterative Algorithm"""

    def __init__(
        self,
        n_iter=100,
        beta=0.8,
        hop_length=None,
        win_length=None,
        n_fft=None,
        window="hann",
        center=True,
        dtype=None,
        length=None,
        pad_mode="constant",
        random_state=None,
    ):
        super().__init__(
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window=window,
            center=center,
            dtype=dtype,
            length=length,
            pad_mode=pad_mode,
            random_state=random_state,
        )

        self.init_random_state(random_state)
        self.beta = beta

    def func_a(self, magspec, stft):
        return (
            self.project_onto_magspec(magspec, stft)
            + (self.project_onto_magspec(magspec, stft) - stft) / self.beta
        )

    def func_c(self, stft):
        return (
            self.project_complex_spec(stft) - (self.project_complex_spec(stft) - stft) / self.beta
        )

    def vocode(self, magspec, init_phase=None, return_phase=False):
        phase = np.empty(magspec.shape, dtype=np.complex64)
        self.eps = librosa.util.tiny(phase)

        if init_phase is None:
            # randomly initialize the phase
            phase[:] = np.exp(2j * np.pi * self.rng.rand(*magspec.shape))
        else:
            phase[:] = init_phase / (np.abs(init_phase) + self.eps)

        # Initialize the complex spectrogram
        stft = magspec * phase

        for _ in range(self.n_iter):
            temp_stft = self.project_complex_spec(
                self.func_a(magspec, stft)
            ) - self.project_onto_magspec(magspec, self.func_c(stft))

            stft[:] = stft + self.beta * temp_stft

        # Return the final phase estimates
        phase[:] = stft / (np.abs(stft) + self.eps)

        gen_audio = librosa.istft(
            magspec * phase,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            dtype=self.dtype,
            length=self.length,
        )

        if return_phase:
            return gen_audio, phase
        return gen_audio


class ADMM(IterVocoder):
    """Alternating Direction Method of Multipliers"""

    def __init__(
        self,
        n_iter=100,
        hop_length=None,
        win_length=None,
        n_fft=None,
        window="hann",
        center=True,
        dtype=None,
        length=None,
        pad_mode="constant",
        random_state=None,
    ):
        super().__init__(
            n_iter=n_iter,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window=window,
            center=center,
            dtype=dtype,
            length=length,
            pad_mode=pad_mode,
            random_state=random_state,
        )

        self.init_random_state(random_state)

    def vocode(self, magspec, init_phase=None, return_phase=False):
        phase = np.empty(magspec.shape, dtype=np.complex64)
        self.eps = librosa.util.tiny(phase)

        if init_phase is None:
            # randomly initialize the phase
            phase[:] = np.exp(2j * np.pi * self.rng.rand(*magspec.shape))
        else:
            phase[:] = init_phase / (np.abs(init_phase) + self.eps)

        # Initialize the complex spectrogram
        stft = magspec * phase

        for _ in range(self.n_iter):
            stft[:] = (
                stft
                + self.project_onto_magspec(magspec, 2 * self.project_complex_spec(stft) - stft)
                - self.project_complex_spec(stft)
            )

        # Return the final phase estimates
        phase[:] = stft / (np.abs(stft) + self.eps)

        gen_audio = librosa.istft(
            magspec * phase,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            dtype=self.dtype,
            length=self.length,
        )

        if return_phase:
            return gen_audio, phase
        return gen_audio


class HybridVocoder(IterVocoder):
    def __init__(self, vocoder_dict, stft_args, random_state=None):
        super().__init__()
        self.init_random_state(random_state)
        self.vocoders = []
        # creates a sequence of vocoders to be used
        for vocoder_name in vocoder_dict:
            kwargs = vocoder_dict[vocoder_name]
            vocoder = self.get_vocoder(vocoder_name)(**kwargs, **stft_args)
            self.vocoders.append(vocoder)

    def get_vocoder(self, vocoder_name):
        """Dispatch vocoder with method"""
        vooder_func = getattr(
            self,
            vocoder_name,
            lambda: "Vocoder not implemented. \
            Choose between: gla, fgla, admm, diffmap, raar",
        )
        return vooder_func

    def gla(self, **kwargs):
        return GriffinLim(**kwargs)

    def fgla(self, **kwargs):
        return FastGriffinLim(**kwargs)

    def admm(self, **kwargs):
        return ADMM(**kwargs)

    def diffmap(self, **kwargs):
        return DiffMap(**kwargs)

    def raar(self, **kwargs):
        return RAAR(**kwargs)

    def vocode(self, magspec, init_phase=None, return_phase=False):
        phase = np.empty(magspec.shape, dtype=np.complex64)
        self.eps = librosa.util.tiny(phase)

        if init_phase is None:
            # randomly initialize the phase
            phase = np.exp(2j * np.pi * self.rng.rand(*magspec.shape))
        else:
            phase = phase / (np.abs(phase) + self.eps)

        for vocoder in self.vocoders:
            _, phase = vocoder.vocode(magspec=magspec, init_phase=phase, return_phase=True)

        gen_audio = librosa.istft(
            magspec * phase,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            dtype=self.dtype,
            length=self.length,
        )

        if return_phase:
            return gen_audio, phase
        return gen_audio
