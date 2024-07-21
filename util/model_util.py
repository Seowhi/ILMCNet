import os
import torch
# from net.model_net import Bert_CRF
import config.args as args


def save_model(model, output_dir, step, acc, loss):
    acc = round(acc, 4)
    loss = round(loss, 4)
    model_to_save = model.module if hasattr(model,
                                            'module') else model    # Only save the model it-self
    output_model_file = os.path.join(output_dir, f"pytorch_model_{acc}_{loss}_{step}.bin")
    torch.save(model_to_save.state_dict(), output_model_file)


def load_model(output_dir):
    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    model_state_dict = torch.load(output_model_file)
    model = Bert_CRF.from_pretrained(args.model, state_dict=model_state_dict)
    return model