"""PromptFirewall — guardrail 2 tầng lọc prompt độc hại trước khi vào LLM.

Sinh ra từ notebook prompt_firewall_toxicchat.ipynb (HSU AI-Driven Challenge 2026).
"""
import re
import json
from pathlib import Path

import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Mẫu tấn công ghi đè chỉ thị / chiếm quyền nhân cách (Phần 7.2 của notebook).
# Nạp tri thức miền vào chỗ dữ liệu huấn luyện không có mẫu.
INJECTION_PATTERNS = [
    r'\b(?:ignore|disregard|forget)\b[^.!?\n]{0,40}\b(?:previous|prior|above|earlier|all)\b'
    r'[^.!?\n]{0,30}\b(?:instruction|prompt|rule|direction|command)',
    r'\b(?:reveal|show|print|repeat|output|tell me|display)\b[^.!?\n]{0,30}'
    r'\b(?:system|initial|original|hidden)\b[^.!?\n]{0,15}\b(?:prompt|instruction|message)',
    r'\b(?:you are|act as|you\'re)\b[^.!?\n]{0,20}\bDAN\b|\bDAN mode\b|\bdo anything now\b',
    r'\b(?:developer mode|jailbreak mode|opposite mode|evil mode|god mode)\b',
    r'\b(?:no|without|free from|removed?|bypass(?:ing)?)\b[^.!?\n]{0,25}'
    r'\b(?:restriction|limitation|filter|censorship|guideline|ethic|moral|rule)s?\b',
    r'\b(?:never|cannot|can\'t|must not|will not)\b[^.!?\n]{0,15}'
    r'\b(?:refuse|decline|say no|reject)\b',
    r'\b(?:unfiltered|uncensored|unrestricted|unlimited)\b[^.!?\n]{0,25}'
    r'\b(?:ai|assistant|model|response|answer|version|chatbot)\b',
]
RX_INJECTION = re.compile('|'.join(f'(?:{p})' for p in INJECTION_PATTERNS), re.I)


class PromptFirewall:
    """Guardrail 2 tầng cho prompt đầu vào LLM.

    Tầng 1: TF-IDF + LinearSVC đã hiệu chỉnh  — chốt phần lớn prompt.
    Tầng 2: DistilBERT multi-label            — chỉ chạy trên dải lưng chừng.
    """

    RISK = ['safe', 'toxic', 'jailbreak', 'injection']

    def __init__(self, vectorizer, fast_clf, bert, tokenizer,
                 lo, hi, th_fast, th_bert, th_jb,
                 max_len=192, batch_size=128, device='cpu', regex_block=True):
        self.vectorizer = vectorizer
        self.fast_clf   = fast_clf
        self.bert       = bert.eval().to(device)
        self.tokenizer  = tokenizer
        self.lo, self.hi = lo, hi
        self.th_fast, self.th_bert, self.th_jb = th_fast, th_bert, th_jb
        self.max_len, self.batch_size, self.device = max_len, batch_size, device
        self.use_amp = (device == 'cuda')
        # regex_block=True  → mẫu tấn công chặn thẳng (bền hơn với dữ liệu ngoài phân bố)
        # regex_block=False → regex chỉ dùng để gán nhãn risk, không đụng quyết định nhị phân
        self.regex_block = regex_block

    # ---------------------------------------------------------- tiền xử lý
    _INV = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]')
    _WS  = re.compile(r'\s+')

    @classmethod
    def _norm(cls, t):
        return cls._WS.sub(' ', cls._INV.sub('', str(t))).strip()

    @staticmethod
    def _regex_hit(texts):
        """Tầng 0 — luật tĩnh, ~0.14s cho 10.000 prompt."""
        return np.array([bool(RX_INJECTION.search(t)) for t in texts], dtype=bool)

    # ---------------------------------------------------------- tầng 2
    @torch.inference_mode()
    def _bert_probs(self, texts):
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_len, padding=False)
        ids = enc['input_ids']
        order = np.argsort([len(x) for x in ids])
        out = np.empty((len(ids), 3), dtype=np.float32)
        pad = self.tokenizer.pad_token_id
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            chunk = [ids[i] for i in idx]
            m = max(len(c) for c in chunk)
            iid = torch.tensor([c + [pad] * (m - len(c)) for c in chunk], device=self.device)
            att = torch.tensor([[1] * len(c) + [0] * (m - len(c)) for c in chunk], device=self.device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.use_amp):
                logits = self.bert(input_ids=iid, attention_mask=att).logits
            out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
        return out

    # ---------------------------------------------------------- API công khai
    def predict(self, prompts):
        """→ np.ndarray[int8]: 0 = safe, 1 = unsafe. Đây là hàm BGK sẽ gọi."""
        if isinstance(prompts, str):
            prompts = [prompts]
        texts = [self._norm(p) for p in prompts]
        p_f = self.fast_clf.predict_proba(self.vectorizer.transform(texts))[:, 1]

        pred = (p_f > self.hi).astype(np.int8)
        band = np.where((p_f >= self.lo) & (p_f <= self.hi))[0]
        if len(band):
            pred[band] = (self._bert_probs([texts[i] for i in band])[:, 0]
                          > self.th_bert).astype(np.int8)
        if self.regex_block:
            pred |= self._regex_hit(texts).astype(np.int8)
        return pred

    def predict_detailed(self, prompts):
        """→ list[dict]: có cả loại rủi ro & độ tin cậy (yêu cầu nâng cao).

        risk ∈ {safe, toxic, jailbreak, injection}: injection = khớp mẫu tấn
        công cấu trúc (RX_INJECTION, vd. "ignore previous instructions",
        "you are DAN"); jailbreak = đầu ra BERT vượt ngưỡng nhưng không khớp
        regex (vd. nhập vai ẩn dụ); toxic = còn lại (bao gồm hate speech).
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        texts = [self._norm(p) for p in prompts]
        p_f = self.fast_clf.predict_proba(self.vectorizer.transform(texts))[:, 1]

        pred = (p_f > self.hi).astype(np.int8)
        conf = np.where(p_f > self.hi, p_f, 1 - p_f)
        p_jb = np.zeros(len(texts), dtype=np.float32)
        stage = np.full(len(texts), 'fast', dtype=object)

        hit = self._regex_hit(texts)

        band = np.where((p_f >= self.lo) & (p_f <= self.hi))[0]
        if len(band):
            P = self._bert_probs([texts[i] for i in band])
            pred[band] = (P[:, 0] > self.th_bert).astype(np.int8)
            conf[band] = np.where(P[:, 0] > self.th_bert, P[:, 0], 1 - P[:, 0])
            p_jb[band] = P[:, 2]
            stage[band] = 'bert'

        if self.regex_block:
            pred = np.maximum(pred, hit.astype(np.int8))
            conf[hit] = 1.0
            stage[hit] = 'regex'

        return [
            {'prompt': (p[:100] + '…') if len(p) > 100 else p,
             'label': 'UNSAFE' if pr else 'SAFE',
             'risk': ('safe' if not pr else
                      ('injection' if h else
                       ('jailbreak' if jb > self.th_jb else 'toxic'))),
             'confidence': round(float(c), 4),
             'stage': st}
            for p, pr, c, jb, st, h in zip(prompts, pred, conf, p_jb, stage, hit)
        ]

    # ---------------------------------------------------------- lưu / nạp
    def save(self, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        joblib.dump({'vectorizer': self.vectorizer, 'fast_clf': self.fast_clf}, path / 'fast.joblib')
        self.bert.save_pretrained(path / 'bert')
        self.tokenizer.save_pretrained(path / 'bert')
        (path / 'config.json').write_text(json.dumps({
            'lo': self.lo, 'hi': self.hi, 'th_fast': self.th_fast,
            'th_bert': self.th_bert, 'th_jb': self.th_jb,
            'max_len': self.max_len, 'batch_size': self.batch_size,
            'regex_block': self.regex_block,
        }, indent=2))
        print(f'đã lưu → {path.resolve()}')

    @classmethod
    def load(cls, path, device='cpu'):
        path = Path(path)
        fast = joblib.load(path / 'fast.joblib')
        cfg = json.loads((path / 'config.json').read_text())
        tok = AutoTokenizer.from_pretrained(path / 'bert')
        mdl = AutoModelForSequenceClassification.from_pretrained(path / 'bert')
        return cls(fast['vectorizer'], fast['fast_clf'], mdl, tok,
                   device=device, **cfg)
