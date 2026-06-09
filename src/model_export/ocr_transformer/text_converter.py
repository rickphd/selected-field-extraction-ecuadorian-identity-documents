import string
charset = string.ascii_uppercase + ' ' + string.digits +"Ñ"
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
    
    def decode(self, indices):
        # devolver como texto el vector de numeros
        chars = []
        for i in indices:
            char = self.idx2char.get(i, "")
            if char == self.EOS_TOKEN:
                break
            if char not in self.special_tokens:
                chars.append(char)
        
        return ''.join(chars)
    
    def vocab_size(self):
        return len(self.tokens)