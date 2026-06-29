import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
        
def has_edit(example):
    return any(
        s["left"] != t["left"] or s["right"] != t["right"] 
        for s, t in zip(example["source"]["base"], example["target"]["local"])
    )

class StructuredTokenDataset(Dataset):
    """
    examples format:

    [
        {
            "source": [
                {"token": "POS", "left": [...], "right": [...]},
                ...
            ],
            "target": {
                "classifier": [
                    {"token": "POS", "left": [...], "right": [...]},
                    ...
                ],
                "decoder": ["(TOP", "deprel", "POS", ")TOP", ...]
            }
        },
        ...
    ]

    source left/right = encoder input structural features
    target["classifier"] left/right = token-level BCE labels
    target["decoder"] = global decoder sequence
    """

    def __init__(self, examples, lang2id, token2id, bracket2id, left2id, right2id):
        self.examples = examples

        self.lang2id = lang2id
        self.token2id = token2id
        self.bracket2id = bracket2id
        self.left2id = left2id
        self.right2id = right2id

        self.lang_unk_id = lang2id["<unk>"]
        self.token_pad_id = token2id["<PAD>"]
        self.token_unk_id = token2id["<UNK>"]

        self.local_decoder_pad_id = left2id["<PAD>"]
        self.local_decoder_bos_id = left2id["<BOS>"]
        self.local_decoder_eos_id = left2id["<EOS>"]
        self.local_decoder_unk_id = left2id.get("<UNK>", self.local_decoder_pad_id)

        self.num_lang = len(lang2id)
        self.num_left = len(left2id)
        self.num_right = len(right2id)

    def __len__(self):
        return len(self.examples)

    def encode_lang(self, lang):
        return self.lang2id.get(lang, self.lang_unk_id)

    def encode_token(self, token):
        return self.token2id.get(token, self.token_unk_id)

    def encode_ids(self, labels, label2id):
        return [label2id[x] for x in labels]
   
    def encode_local_decoder_sequence(self, seq, label2id):
        ids = self.encode_ids(seq, label2id)

        dec_input_ids = [self.local_decoder_bos_id] + ids
        dec_target_ids = ids + [self.local_decoder_eos_id]

        return dec_input_ids, dec_target_ids

    def build_gate_label(self, src_item, tgt_item):
        """
        1 = needs correction
        0 = keep source / baseline
        """
        return int(
            src_item["left"] != tgt_item["left"]
            or src_item["right"] != tgt_item["right"]
        )

    def __getitem__(self, idx):
        ex = self.examples[idx]

        lang = ex["lang"]
        source = ex["source"]
        source_base = source["base"]
        source_overlap = source["overlap"]
        decoder_target = ex["target"]["local"]

        assert len(source_base) == len(decoder_target), (
            f"Length mismatch at idx={idx}: "
            f"source={len(source)}, decoder={len(decoder_target)}"
        )
       
        input_ids = []
        base_left_ids = []
        base_right_ids = []
        overlap_left_ids = []
        overlap_right_ids = []

        left_dec_input_ids = []
        right_dec_input_ids = []

        left_dec_target_ids = []
        right_dec_target_ids = []

        gate_targets = []

        raw_source = []
        raw_local_decoder_target = []

        lang_id = self.encode_lang(lang)

        for i, (sbase, soverlap, t) in enumerate(zip(source_base, source_overlap, decoder_target)):
            assert sbase["token"] == t["token"], (
                f"Token mismatch at idx={idx}, pos={i}: "
                f"source={s['token']!r}, target={t['token']!r}"
            )

            token = sbase["token"]

            input_ids.append(self.encode_token(token))

            # source structural features
            base_left_ids.append(self.encode_ids(sbase["left"], self.left2id))
            base_right_ids.append(self.encode_ids(sbase["right"], self.right2id))

            # overlapping left and right
            overlap_left_ids.append(self.encode_ids(soverlap["left"], self.bracket2id))
            overlap_right_ids.append(self.encode_ids(soverlap["right"], self.bracket2id))
            
            # local decoder input and targets
            token_left_dec_input_ids, token_left_dec_target_ids = self.encode_local_decoder_sequence(
                t["left"],
                self.left2id
            )
            token_right_dec_input_ids, token_right_dec_target_ids = self.encode_local_decoder_sequence(
                t["right"],
                self.right2id
            )
            
            left_dec_input_ids.append(token_left_dec_input_ids)
            left_dec_target_ids.append(token_left_dec_target_ids)
            right_dec_input_ids.append(token_right_dec_input_ids)
            right_dec_target_ids.append(token_right_dec_target_ids)

            gate_targets.append(self.build_gate_label(sbase, t))

            raw_source.append(sbase)
            raw_local_decoder_target.append(t)

        return {
            "lang_id": lang_id,
            "input_ids": input_ids,

            # source structural features
            "base_left_ids": base_left_ids,
            "base_right_ids": base_right_ids,

            "overlap_left_ids": overlap_left_ids,
            "overlap_right_ids": overlap_right_ids,

            # local decoder CE labels
            "left_dec_input_ids": left_dec_input_ids,
            "left_dec_target_ids": left_dec_target_ids,
            "right_dec_input_ids": right_dec_input_ids,
            "right_dec_target_ids": right_dec_target_ids,
            
            # gate targets
            "gate_targets": gate_targets,

            # raw debug info
            "source": raw_source,
            "local_decoder_target": raw_local_decoder_target,
        }

def fake_dec_seq(max_s, pad_id, bos_id, eos_id):
    """
    Avoid all <PAD>
    """
    if max_s <= 1:
        return [bos_id]
    return [bos_id, eos_id] + [pad_id] * (max_s - 2)

def structured_token_collate_fn(batch, token_pad_id=0, decoder_pad_id=0, decoder_bos_id=1, decoder_eos_id=2):
    max_len = max(len(item["input_ids"]) for item in batch)

    lang_ids = []
    input_ids = []
    attention_mask = []

    base_left_ids = []
    base_right_ids = []

    overlap_left_ids = []
    overlap_right_ids = []

    left_dec_input_ids = []
    left_dec_target_ids = []
    right_dec_input_ids = []
    right_dec_target_ids = []

    gate_targets = []

    raw_source = []
    raw_local_decoder_target = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        lang_ids.append(item["lang_id"])
        input_ids.append(item["input_ids"] + [token_pad_id] * pad_len)
        attention_mask.append([1] * seq_len + [0] * pad_len)

        # keep nested lists because each token has variable number of source features
        base_left_ids.append(item["base_left_ids"] + [[] for _ in range(pad_len)])
        base_right_ids.append(item["base_right_ids"] + [[] for _ in range(pad_len)])

        overlap_left_ids.append(item["overlap_left_ids"] + [[] for _ in range(pad_len)])
        overlap_right_ids.append(item["overlap_right_ids"] + [[] for _ in range(pad_len)])
        

        max_left_input = max(len(seq) for item in batch for seq in item["left_dec_input_ids"]) 
        max_right_input = max(len(seq) for item in batch for seq in item["right_dec_input_ids"])

        max_left_target = max(len(seq) for item in batch for seq in item["left_dec_target_ids"])
        max_right_target = max(len(seq) for item in batch for seq in item["right_dec_target_ids"])

        padded_left_inputs = [
            seq + [decoder_pad_id] * (max_left_input - len(seq)) 
            for seq in item["left_dec_input_ids"]
        ]
        padded_right_inputs = [
            seq + [decoder_pad_id] * (max_right_input - len(seq)) 
            for seq in item["right_dec_input_ids"]
        ]

        padded_left_targets = [
            seq + [decoder_pad_id] * (max_left_target - len(seq)) 
            for seq in item["left_dec_target_ids"]
        ]
        padded_right_targets = [
            seq + [decoder_pad_id] * (max_right_target - len(seq)) 
            for seq in item["right_dec_target_ids"]
        ]

        left_dec_input_ids.append(
            padded_left_inputs +  
            [fake_dec_seq(max_left_input, decoder_pad_id, decoder_bos_id, decoder_eos_id)
             for _ in range(pad_len)]
            
        )
        right_dec_input_ids.append(
            padded_right_inputs +  
            [fake_dec_seq(max_right_input, decoder_pad_id, decoder_bos_id, decoder_eos_id)
             for _ in range(pad_len)]
        )

        left_dec_target_ids.append(padded_left_targets + [[decoder_pad_id] * max_left_target for _ in range(pad_len)])
        right_dec_target_ids.append(padded_right_targets + [[decoder_pad_id] * max_right_target for _ in range(pad_len)])


        gate_targets.append(item["gate_targets"] + [0] * pad_len)

        raw_source.append(item["source"])
        raw_local_decoder_target.append(item["local_decoder_target"])

    return {
        "lang_ids": torch.tensor(lang_ids, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),

        "base_left_ids": base_left_ids,
        "base_right_ids": base_right_ids,

        "overlap_left_ids": overlap_left_ids,
        "overlap_right_ids": overlap_right_ids,

        "left_dec_input_ids": torch.tensor(left_dec_input_ids, dtype=torch.long),
        "left_dec_target_ids": torch.tensor(left_dec_target_ids, dtype=torch.long),
        "right_dec_input_ids": torch.tensor(right_dec_input_ids, dtype=torch.long),
        "right_dec_target_ids": torch.tensor(right_dec_target_ids, dtype=torch.long),

        "gate_targets": torch.tensor(gate_targets, dtype=torch.float),

        "source": raw_source,
        "local_decoder_target": raw_local_decoder_target,
    }

def make_dataloader(examples, lang2id, token2id, bracket2id, left2id, right2id, batch_size=8, shuffle=False):
    dataset = StructuredTokenDataset(
        examples=examples,
        lang2id=lang2id,
        token2id=token2id,
        bracket2id=bracket2id,
        left2id=left2id,
        right2id=right2id,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: structured_token_collate_fn(
            batch,
            token_pad_id=token2id["<PAD>"],
            decoder_pad_id=left2id["<PAD>"],
        ),
    )
