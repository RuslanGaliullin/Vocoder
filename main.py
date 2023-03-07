from scipy.io import wavfile as wav
from scipy.fftpack import fft, ifft
import sys
from scipy import signal
import numpy as np


class Vocoder:

    # Инициализация начальных параметров для переданного аудио
    def __init__(self, rate: int, data: np.ndarray) -> None:
        self._sample_rate = rate
        self._data = data
        self._hop_a = 512

        # Размер фрейма
        self._frame_size = self._hop_a * 4

        # Промежуток времени между началами взятых фреймов
        self._delta_t_a = self._hop_a / rate
        self._frames = []

        # Искомые частоты в звуке
        self._frequency_bins = np.fft.fftfreq(self._frame_size, d=1 / self._frame_size)

    # Анализ каждого фрейма с помощью FFT и Hanning окна
    def __analyse(self) -> None:
        self._frames = []
        for i in range(0, len(self._data) - self._frame_size + 1, self._hop_a):
            self._frames.append(fft(self._data[i:i + self._frame_size] * signal.windows.hann(self._frame_size)))

    # Преобразование фаз и частот между фреймами
    def __processing(self, delta_t_s: float) -> list[np.ndarray]:
        X_real = np.abs(self._frames)
        X_phase = np.angle(self._frames)

        # Итоговый список фреймов со смещенными фазами
        X_s = [self._frames[0]]
        for frame in range(1, len(self._frames)):
            delta_omega = (X_phase[frame] - X_phase[frame - 1]) / self._delta_t_a - self._frequency_bins
            delta_wrapped_omega = (delta_omega + np.pi) % (2 * np.pi) - np.pi
            true_omega = self._frequency_bins + delta_wrapped_omega
            X_s_phase = np.angle(X_s[frame - 1]) + delta_t_s * true_omega
            X_s.append(X_real[frame] * np.cos(X_s_phase) + X_real[frame] * np.sin(X_s_phase) * 1j)
        return X_s

    # Применение обратного FFT к фреймам и запись в результирующий массив с амплитудами итогового аудио
    def __synthesis(self, X_s: list[np.ndarray], hop_s: int) -> np.ndarray[np.float32]:
        q = []
        for frame in X_s:
            q.append(np.real(ifft(frame) * signal.windows.hann(self._frame_size)).astype(np.float32))

        # Наложение фреймов с учетом сдвига на hop_s семплов
        result_samples = []
        for i in range(len(X_s)):
            for j in range(self._frame_size):
                if i * hop_s + j >= len(result_samples):
                    result_samples.append(q[i][j])
                else:
                    result_samples[i * hop_s + j] += q[i][j]

        return np.array(result_samples)

    # Handler для запуска анализа, процессинга и синтеза для указанного соотношения сжатия/растяжения
    def convert(self, scale: float) -> np.ndarray[np.float32]:
        hop_s = int(self._hop_a * scale)
        delta_t_s = hop_s / self._sample_rate

        self.__analyse()
        X_s = self.__processing(delta_t_s)
        y = self.__synthesis(X_s, hop_s)

        return y / np.max(y)


if __name__ == '__main__':
    # Считываем файл
    sample_rate, samples = wav.read(sys.argv[1])
    print(len(samples))
    # Определяем коэффициент hop_s/hop_a
    coefficient = 2 ** ((float(sys.argv[3]) < 1) * (-1) + (float(sys.argv[3]) >= 1))
    coder = Vocoder(sample_rate, samples)
    result = coder.convert(coefficient)
    print(len(result))
    wav.write(sys.argv[2], sample_rate, result)
