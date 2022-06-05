from sacred import Experiment 

ex = Experiment("explanation-generation")

@ex.config 
def config():
    seed = 12345
    run_name = "roberta-spurious-sst-finetune"
    # Model params (set later)
    pretrained_model_name = None
    pretrained_model_config = None
    tokenizer_config_name = None
    # dataset params (set later)
    dataset_name = None
    dataset_kwargs = None
    num_labels = None 
    batch_size = 32
    report_to = "wandb"
    # report_to = "none"
