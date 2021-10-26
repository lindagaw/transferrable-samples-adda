"""Main script for ADDA."""
import pretty_errors
import params

from torchvision import datasets, transforms, models
import torch
import torch.nn as nn

from core import eval_src, eval_tgt, train_src, train_tgt
from core import generate
from models import Discriminator, LeNetClassifier, LeNetEncoder, Generator
from utils import get_data_loader, init_model, init_random_seed

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_data_loader = get_data_loader(params.src_dataset)
    src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    tgt_data_loader = get_data_loader(params.tgt_dataset)
    tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models
    progenitor = models.resnet50(pretrained=True)
    progenitor.fc = torch.nn.Linear(2048, 10)
    progenitor = progenitor.to(torch.device('cuda:0'))

    src_encoder = torch.nn.Sequential(*(list(progenitor.children())[0:-1])).to(torch.device('cuda:0'))
    tgt_encoder = torch.nn.Sequential(*(list(progenitor.children())[0:-1])).to(torch.device('cuda:0'))
    src_classifier = torch.nn.Linear(2048, 10).to(torch.device('cuda:0'))

    critic = models.resnet50(pretrained=True)
    critic.fc = torch.nn.Linear(2048, 2)
    critic = critic.to(torch.device('cuda:0'))

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)


    src_encoder, src_classifier = train_src(src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    generator = init_model(Geneator(input_length=28),
                        restore='')

    generator, critic = generate(generator, critic,
                  src_data_loader, tgt_data_loader)

    # init weights of target encoder with those of source encoder
    tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
