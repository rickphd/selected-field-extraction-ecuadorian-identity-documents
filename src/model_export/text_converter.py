import string
import numpy as np
charset = string.ascii_uppercase + ' ' + string.digits + "Ñ"
class TextConverter:
    def __init__(self, charset=charset):
        self.charset = charset
        self.SOS_TOKEN = '<SOS>' # Start of Sentence
        self.EOS_TOKEN = '<EOS>' # End of Sentence
        self.PAD_TOKEN = '<PAD>' # Placeholder - igualar las dimensiones
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        self.tokens = self.special_tokens + list(charset)

        self.char2idx = {char: idx for idx, char in enumerate(self.tokens)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
    
    def encode(self, text, max_len=None):
        # devolver como vector de numeros el texto
        tokens = [self.char2idx[self.SOS_TOKEN]] + \
                 [self.char2idx[c] for c in text] + \
                 [self.char2idx[self.EOS_TOKEN]]

        if max_len is not None:
            tokens = tokens[:max_len]
            pad_len = max_len - len(tokens)
            tokens += [self.char2idx[self.PAD_TOKEN]] * max(0, pad_len)
        
        return tokens

    def decode(self, logits: np.ndarray, softmax_probs=False):
        """
        Decodifica la salida del modelo OCR (por ejemplo, (1, 25, 41)).
        Primero detecta el rango SOS->EOS y luego aplica softmax solo sobre ese rango
        para calcular la probabilidad promedio.

        Args:
            logits (np.ndarray): Salida del modelo (1, T, C) o (T, C).
            softmax_probs (bool): Si True, devuelve (texto, probabilidad_promedio entre SOS->EOS).

        Returns:
            str o (str, float): texto o (texto, confianza_promedio SOS->EOS)
        """

        # --- Eliminar batch si existe ---
        if logits.ndim == 3 and logits.shape[0] == 1:
            logits = logits[0]  # (T, C)
        elif logits.ndim != 2:
            raise ValueError(f"Forma inesperada de logits: {logits.shape}")

        # --- Obtener índices predichos ---
        indices = np.argmax(logits, axis=1)  # (T,)

        # --- Buscar posiciones de SOS y EOS ---
        sos_idx = None
        eos_idx = None

        for t, i in enumerate(indices):
            char = self.idx2char.get(int(i), "")
            if char == self.SOS_TOKEN and sos_idx is None:
                sos_idx = t
            elif char == self.EOS_TOKEN and sos_idx is not None:
                eos_idx = t
                break

        # Si no hay SOS o EOS, devolver vacío o todo
        if sos_idx is None or eos_idx is None or eos_idx <= sos_idx:
            if softmax_probs:
                return "", 0.0
            return ""

        # --- Extraer rango de interés ---
        indices_range = indices[sos_idx + 1 : eos_idx]  # caracteres entre SOS y EOS
        logits_range = logits[sos_idx : eos_idx + 1]    # incluye SOS y EOS para prob

        # --- Decodificar caracteres ---
        chars = []
        for i in indices_range:
            char = self.idx2char.get(int(i), "")
            if char not in self.special_tokens:
                chars.append(char)
        text = ''.join(chars)

        # --- Calcular probabilidad promedio solo si se solicita ---
        if not softmax_probs:
            return text

        # Aplicar softmax SOLO en el rango SOS→EOS
        exp_logits = np.exp(logits_range - np.max(logits_range, axis=1, keepdims=True))
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        probs = np.max(softmax, axis=1)  # (EOS - SOS + 1,)

        avg_precision = float(np.mean(probs)) if probs.size > 0 else 0.0
        return text, avg_precision

    
    def vocab_size(self):
        return len(self.tokens)