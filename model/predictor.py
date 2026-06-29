import json
import torch
from pathlib import Path
from tqdm import tqdm
from model.generation import generate_local_sequence

class StructuredTokenPredictor:
    def __init__(
        self, 
        decoder,
        decoder_checkpoint,
        device,
        id2left,
        id2right,
        max_left_len,
        max_right_len,
        gate_threshold=0.8,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        gate=None,
        gate_checkpoint=None,
    ):
        self.device = device
        self.decoder = self.load_checkpoint_from_pt(decoder, decoder_checkpoint)
        
        if gate is not None and gate_checkpoint is not None:
            self.gate = self.load_checkpoint_from_pt(gate, gate_checkpoint)
        else:
            self.gate = None

        self.id2left = id2left
        self.id2right = id2right

        self.max_left_len = max_left_len
        self.max_right_len = max_right_len
        
        self.gate_threshold = gate_threshold

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        
    def load_checkpoint_from_pt(self, model, checkpoint):
        state_dict = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(state_dict["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def generate_local_sequence(self, batch, side, max_len):
        return generate_local_sequence(
            model=self.decoder,
            batch=batch,
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            device=self.device,
            side=side,
            max_len=max_len,
        )

    @torch.no_grad()
    def predict_correction(self, batch):
        lang_ids = batch["lang_ids"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        base_left_ids = batch["base_left_ids"]
        base_right_ids = batch["base_right_ids"]

        overlap_left_ids = batch["overlap_left_ids"]
        overlap_right_ids = batch["overlap_right_ids"]
    
        outputs = self.gate(
            lang_ids=lang_ids,
            input_ids=input_ids,
            base_left_ids=base_left_ids,
            base_right_ids=base_right_ids,
            overlap_left_ids=overlap_left_ids,
            overlap_right_ids=overlap_right_ids,
            attention_mask=attention_mask,

        )
    
        gate_prob = outputs["gate_prob"]
        use_model = (gate_prob > self.gate_threshold) & (attention_mask.bool())
    
        return {
            "gate_prob": gate_prob,
            "use_model": use_model,
        }
    
    def ids_to_labels(self, ids, id2label):
        labels = []
        for x in ids:
            # print(ids)
            x = int(x)
            if x in (self.pad_id, self.bos_id):
                continue
            if x == self.eos_id:
                break
            labels.append(id2label[str(x)])
        return labels
    

    def merge_decoder_and_gate_outputs(self, pred_ids, baseline_ids, id2label, use_model):
        B, T = use_model.shape
        final = []
    
        for b in range(B):
            sent = []
            for t in range(T):
                if use_model[b, t].item(): 
                    seq = self.ids_to_labels(pred_ids[b, t], id2label)
                    # print(seq)
                else:
                    seq = self.ids_to_labels(baseline_ids[b][t], id2label)
                sent.append(seq)
               
            final.append(sent)
        return final

    def predict(self, dataloader, save_dir: str):
        outputs = []
        
        for batch in tqdm(dataloader, desc="Predicting"):
            
            batch_source = batch["source"]
            left_ids = batch["base_left_ids"]
            right_ids = batch["base_right_ids"]
        
            pred_left = self.generate_local_sequence(batch, side="left", max_len=self.max_left_len)
            pred_right = self.generate_local_sequence(batch, side="right",max_len=self.max_right_len)

            if self.gate is not None:
                use_model = self.predict_correction(batch)["use_model"]
            else:
                use_model = batch["attention_mask"]
    
            final_left_labels = self.merge_decoder_and_gate_outputs(
                pred_ids=pred_left, 
                baseline_ids=left_ids, 
                id2label=self.id2left,
                use_model=use_model
            )
    
            final_right_labels = self.merge_decoder_and_gate_outputs(
                pred_ids=pred_right, 
                baseline_ids=right_ids, 
                id2label=self.id2right,
                use_model=use_model
            )
    
        
            for sent_src, sent_left, sent_right in zip(batch_source, final_left_labels, final_right_labels):
                sent_out = []
                for src_item, left, right in zip(sent_src, sent_left, sent_right):
                    sent_out.append({
                        "token": src_item["token"],
                        "left": left,
                        "right": right,
                    })
                outputs.append(sent_out)
        

        if save_dir is not None:
            with open(Path(save_dir), "w", encoding="utf-8") as fout:
                json.dump(outputs, fout, indent=4)
                
        return outputs
    