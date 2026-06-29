import torch
from pathlib import Path
from model.load_data import read_jsonl, make_dataloader
from model.decoder import StructuredTokenDecoder
from model.gate import StructuredTokenGate
from model.predictor import StructuredTokenPredictor

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "downstream"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "[v3.0]structured_token_edit"
PRED_DIR = BASE_DIR / "predictions" / "downstream" 

lang = "en-penn"

decoder_config_path = ARTIFACT_DIR / f"{lang}_decoder_config.json"
decoder_config = read_jsonl(decoder_config_path)["model"]
decoder_checkpoint = ARTIFACT_DIR / f"{lang}_decoder.pt"

gate_config_path = ARTIFACT_DIR / "gate_config.json"
gate_config = read_jsonl(gate_config_path)["model"]
gate_checkpoint = ARTIFACT_DIR / "gate.pt"


vocab_and_weight_path = ARTIFACT_DIR / "vocab_and_weight.json"
vocab_and_weight = read_jsonl(vocab_and_weight_path)
lang2id = vocab_and_weight["lang2id"]
token2id = vocab_and_weight["token2id"] 
bracket2id = vocab_and_weight["bracket2id"] 
left2id = vocab_and_weight["left2id"]
right2id = vocab_and_weight["right2id"]
id2left = vocab_and_weight["id2left"]
id2right = vocab_and_weight["id2right"]

data = DATA_DIR / f"lang={lang},pos=upos,overlap=3"
test_path = data / "test.json"
test_examples = read_jsonl(test_path)

batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataloader = make_dataloader(
    test_examples,
    lang2id,
    token2id,
    bracket2id,
    left2id,
    right2id,
    batch_size=batch_size,
    shuffle=False,
    )

decoder = StructuredTokenDecoder(**decoder_config, device=device)
gate = StructuredTokenGate(**gate_config, device=device)


predictor = StructuredTokenPredictor(
    decoder=decoder,
    decoder_checkpoint=decoder_checkpoint,
    device=device,
    id2left=id2left,
    id2right=id2right,
    max_left_len=10,
    max_right_len=10,
    gate_threshold=0.3,
    pad_id=left2id["<PAD>"],
    bos_id=left2id["<BOS>"],
    eos_id=left2id["<EOS>"],
    # gate=gate,
    # gate_checkpoint=gate_checkpoint,
    )

predictor.predict(
    dataloader=test_dataloader,
    save_dir=str(PRED_DIR / f"lang={lang},pos=upos,gate=yes.json"),
)

    
    