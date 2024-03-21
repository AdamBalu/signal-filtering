# # imports
import wave
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk


def dft(vector):
    # convert to float if it's not already
    vector = np.asarray(vector, dtype=float)
    N = len(vector)

    # n = [a1, b1, c1..]
    n = np.arange(N)
    # k = [[a2], [b2], [c2]..]
    k = n.reshape((N, 1))

    # grid = [[a1*a2, a1*b2, a1*c2], [b1*a2, b1*b2, b1*c2],...]
    grid = n * k

    # e^(-2jπ * G / N)       ##for each G - grid element
    matrix = np.exp(-2j * np.pi * grid / N)

    # matrix multiplication
    return np.dot(matrix, vector)


def showcase_dft(fs, matrix):
    x = np.arange(0, fs / 2, fs / 2 / 1024)
    y = np.abs(dft(matrix[5]))
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.plot(x, y)
    plt.gca().set_title("Diskrétna fourierova transformácia")
    plt.xlabel("Frekvencia (Hz)")
    plt.show()
    return


def plot_spectrogram(f, t, sgr):
    sgr_log = 10 * np.log10(sgr ** 2 + 1e-20)
    plt.figure(figsize=(7, 5))
    plt.pcolormesh(t, f, sgr_log)
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralni hustota vykonu [dB]', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.show()
    return


def own_signal(duration, fs, frames, signal):
    f1 = 550
    f2 = 1100
    f3 = 1650
    f4 = 2200
    plt.xlabel('Čas (s)')
    plt.ylabel('Fs')

    time = []
    # avg = sum(signal_all) / len(signal_all)
    # max_abs = abs(max(max(signal_all), min(signal_all)))
    # float_sig = signal_all.astype(float)

    for i in range(frames):
        time.append(i/fs)

    signal1 = np.cos(2*np.pi*f1*np.array(time))
    signal2 = np.cos(2*np.pi*f2*np.array(time))
    signal3 = np.cos(2*np.pi*f3*np.array(time))
    signal4 = np.cos(2*np.pi*f4*np.array(time))

    signal_all = signal1 + signal2 + signal3 + signal4

    # plt.plot(signal_all)
    # plt.show()

    signal_a_c = signal_all.copy()
    avg = sum(signal_a_c) / len(signal_a_c)
    max_abs = abs(max(max(signal_a_c), min(signal_a_c)))
    float_sig = signal_a_c.astype(float)
    for i in range(len(signal_a_c)):
        float_sig[i] -= avg
        float_sig[i] /= max_abs
    f, t, sgr = spectrogram(float_sig, fs)
    plot_spectrogram(f, t, sgr)

    # wavfile.write("audio/4cos.wav", int(len(float_sig)/duration), float_sig.astype(np.float32))
    wavfile.write('../audio/4cos.wav', int(fs), (float_sig * np.iinfo(np.int16).max).astype(np.int16))

    filter_main_signal(duration, signal, f1, f2, f3, f4, fs)
    return


def filter_main_signal(duration, signal, f1, f2, f3, f4, fs):
    signal_final = bandstop_filters(signal, f1, fs)
    signal_final = bandstop_filters(signal_final, f2, fs)
    signal_final = bandstop_filters(signal_final, f3, fs)
    signal_final = bandstop_filters(signal_final, f4, fs)
    wavfile.write('../audio/clean_bandstop.wav', int(fs), (signal_final * np.iinfo(np.int16).max).astype(np.int16))
    # plt.plot(signal_final)
    # plt.show()
    return


def bandstop_filters(main_signal, f, fs):
    Q = 30.0  # quality factor
    b, a = scipy.signal.iirnotch(f, Q, fs)
    freq, h = scipy.signal.freqz(b, a, fs=fs)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    output_signal = scipy.signal.filtfilt(b, a, main_signal)

    ir_fchar_zeropts(a, b, fs)  # <-- comment this for filters with frequencies graphs (also uncomment line 126)

    ax[0].plot(freq, 20 * np.log10(abs(h)), color='blue')
    ax[0].set_title("Filtr" + " pro f = " + str(f) + "Hz")
    ax[0].set_ylabel("Amplituda (dB)", color='blue')
    # plt.show()  # <-- uncomment this for filters with frequencies graphs
    return output_signal


def ir_fchar_zeropts(a, b, fs):
    N_imp = 35
    imp = [1, *np.zeros(N_imp - 1)]
    h = lfilter(b, a, imp)

    w, H = freqz(b, a)
    z, p, k = tf2zpk(b, a)

    _, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(w/2/np.pi * fs, np.abs(H))
    ax[0].set_xlabel('Frekvence [Hz]')
    ax[0].set_title('Modul frekvenční charakteristiky $|H(e^{j\omega})|$')

    ax[1].plot(w / 2 / np.pi * fs, np.angle(H))
    ax[1].set_xlabel('Frekvence [Hz]')
    ax[1].set_title('Argument frekvenční charakteristiky $\mathrm{arg}\ H(e^{j\omega})$')

    for ax1 in ax:
        ax1.grid(alpha=0.5, linestyle='--')

    plt.tight_layout()

    plt.figure(figsize=(4, 3.5))

    # jednotkova kruznice
    ang = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(ang), np.sin(ang))

    # nuly, poly
    plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')

    plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
    plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(loc='upper right')

    plt.tight_layout()

    plt.figure(figsize=(5, 3))
    plt.stem(np.arange(N_imp), h, basefmt=' ')
    plt.gca().set_xlabel('$n$')
    plt.gca().set_title('Impulsni odezva $h[n]$')
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()
    return


def visualize(path):
    # reading the audio file
    raw = wave.open(path)

    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype="int16")

    frames = raw.getnframes()
    rate = raw.getframerate()
    duration = frames / rate

    signal2 = signal.copy()
    avg = sum(signal) / len(signal)
    max_abs = abs(max(max(signal), min(signal)))
    float_sig = signal2.astype(float)

    for i in range(len(signal2)):
        float_sig[i] -= avg
        float_sig[i] /= max_abs

    index = 0
    inside_i = 1
    l = []
    matrix = []
    while index < len(float_sig):
        l.append(float_sig[index])
        if inside_i == 1024:
            inside_i = 0
            index -= 512
            matrix.append(l.copy())
            l = []
        inside_i += 1
        index += 1

    matrix = np.array([np.array(xi) for xi in matrix])

    # gets the frame rate
    f_rate = raw.getframerate()

    time = np.linspace(
        0,  # start
        len(signal) / f_rate,
        num=len(signal)
    )

    matrix_chop = matrix[25]
    time2 = np.linspace(0, len(matrix_chop) / f_rate, num=len(matrix_chop))

    # plt.figure(1)

    # plt.title("Sound Wave")
    # plt.xlabel("Time")

    # actual plotting
    # plt.plot(time, float_sig)
    # plt.plot(time2, matrix[34])

    # premenna pre porovnanie funkcionality vlastnej dft s kniznicovou fft
    # comparison = np.allclose(dft(matrix[34]), np.fft.fft(matrix[34]))

    #######################
    # DFT showcase
    #######################

    fs = len(signal) / duration
    showcase_dft(fs, matrix)

    # plt.plot(time2, dft(matrix[5]))
    # shows the plot
    # in new window
    # plt.show()

    f, t, sgr = spectrogram(signal, fs, nperseg=1024, noverlap=512)
    plot_spectrogram(f, t, sgr)

    own_signal(duration, fs, frames, float_sig)
    return


if __name__ == "__main__":
    visualize("xbalus01.wav")
