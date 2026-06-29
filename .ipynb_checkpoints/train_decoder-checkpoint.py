from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.load_data import read_jsonl, has_edit, StructuredTokenDataset, structured_token_collate_fn, make_dataloader
from model.decoder import StructuredTokenDecoder
from model.decoder_trainer import DecoderTrainer

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "downstream"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "[v3.0]structured_token_edit"


def main():
    lang = "en-penn"
    data = DATA_DIR / f"lang={lang},pos=upos,overlap=3"
    train_path = data / "train.json"
    dev_path = data / "dev.json"
    vocab_path = ARTIFACT_DIR / "vocab_and_weight.json"

    artifacts = read_jsonl(vocab_path)

    lang2id = artifacts["lang2id"]
    token2id = artifacts["token2id"]
    bracket2id = artifacts["bracket2id"]
    left2id = artifacts["left2id"]
    right2id = artifacts["right2id"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    left_class_weight = torch.tensor(
        artifacts["left_class_weight"],
        dtype=torch.float,
        device=device,
    )
    right_class_weight = torch.tensor(
        artifacts["right_class_weight"],
        dtype=torch.float,
        device=device,
    )


    train_examples = read_jsonl(train_path)
    dev_examples = read_jsonl(dev_path)

    batch_size = 8

    train_edit_examples = [ex for ex in train_examples if has_edit(ex)]
    train_no_edit_examples = [ex for ex in train_examples if not has_edit(ex)]

    edit_dataset = StructuredTokenDataset(
        examples=train_edit_examples,
        lang2id=lang2id,
        token2id=token2id,
        bracket2id=bracket2id,
        left2id=left2id,
        right2id=right2id,
    )

    no_edit_dataset = StructuredTokenDataset(
        examples=train_no_edit_examples,
        lang2id=lang2id,
        token2id=token2id,
        bracket2id=bracket2id,
        left2id=left2id,
        right2id=right2id,
    )

    # train_loader = make_dataloader(
    #     train_edit_examples,
    #     token2id,
    #     left2id,
    #     right2id,
    #     batch_size=batch_size,
    #     shuffle=True,
    # )

    dev_edit_examples = [ex for ex in dev_examples if has_edit(ex)]
    dev_no_edit_examples = [ex for ex in dev_examples if not has_edit(ex)]
    
    dev_loader = make_dataloader(
        dev_edit_examples,
        lang2id,
        token2id,
        bracket2id,
        left2id,
        right2id,
        batch_size=batch_size,
        shuffle=False,
    )
    
    no_edit_sample_mode = "relative_to_edit"
    # no_edit_schedule = [
    #     (1, 0.0),
    #     (20, 1.0),
    #     (40, 2.0),
    #     (60, 5.0),
    #     (80, None),
    # ]

    no_edit_schedule = [
        (1, None),
    ]

    model_config = {
        "num_langs": len(lang2id),
        "vocab_size": len(token2id),
        "num_total_labels": len(bracket2id),
        "num_left_labels": len(left2id),
        "num_right_labels": len(right2id),
        "device": device,
        "d_model": 512,   # 256 -> 384
        "nhead": 8,       # 4 -> 6
        "num_layers": 6,
        "decoder_layers": 2,  # 1 -> 2
        "dim_feedforward": 512,
        "dropout": 0.0,
        "pad_token_id": token2id["<PAD>"],
        "pad_bracket_id": left2id["<PAD>"],
        "bos_bracket_id": left2id["<BOS>"],
        "eos_bracket_id": left2id["<EOS>"],
        "max_len": 2048,
    }

    model = StructuredTokenDecoder(**model_config)

    trainer = DecoderTrainer(
        lang=lang,
        model=model,
        # train_loader=train_loader,
        edit_dataset=edit_dataset,
        no_edit_dataset=no_edit_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: structured_token_collate_fn(
            batch,
            token_pad_id=token2id["<PAD>"],
            decoder_pad_id=left2id["<PAD>"],
        ),
        no_edit_schedule=no_edit_schedule,
        no_edit_sample_mode=no_edit_sample_mode,
        # no_edit_ratio=None,
        # warmup_epochs=0,
        dev_loader=dev_loader,
        token2id=token2id,
        left2id=left2id,
        right2id=right2id,
        left_class_weight=left_class_weight,
        right_class_weight=right_class_weight,
        device=device,
        # lr=3e-4,
        lr=1e-4,
        weight_decay=1e-2,
        grad_clip=1.0,
        left_dec_loss_weight=1.0,
        right_dec_loss_weight=1.0,
        structure_loss_weight=0.0,
    )

    trainer.fit(
        num_epochs=20,
        lr_milestones=[5, 10, 15],
        save_dir=str(ARTIFACT_DIR),
        save_best_only=True,
    )


if __name__ == "__main__":
    main()