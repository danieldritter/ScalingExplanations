from transformers import Trainer 
import torch 


"""
Currently not used. Have to find a way to prevent spamming accuracy values to console, plus it's less extensible than
the callback version that handles all metrics. Plenty faster if we care about training metrics beyond loss though. 
"""
class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        MAX: Subclassed to compute training accuracy.

        How the loss is computed by Trainer. By default, all models return the loss in
        the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        # MAX: Start of new stuff.
        if "label" in inputs or "labels" in inputs:
            preds = outputs.logits.detach()
            acc = (
                (preds.argmax(axis=-1) == inputs["labels"])
                .type(torch.float)
                .mean()
                .item()
            )
            self.log({"train_accuracy": acc})

        # MAX: End of new stuff.

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss