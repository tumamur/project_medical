import argparse
from models.imagen_pytorch.imagen_pytorch import Unet, Imagen
from models.imagen_pytorch.trainer import ImagenTrainer
from models.imagen_pytorch.data import NLMCXRDataset
import os
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an imagen model')
    parser.add_argument('--iterations', type=int, default=100000, help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--image_path', type=str, default='/home/guo/git/Rad-ReStruct/data/radrestruct/images/', help='path to images')
    parser.add_argument('--text_path', type=str, default='/home/guo/data/ecgen-radiology/', help='path to text')
    parser.add_argument('--model_path', type=str, default='', help='path to resume model')
    parser.add_argument('--load_model', type=bool, default=False, help='resume training')
    parser.add_argument('--save_path', type=str, default='..', help='path to save checkpoints and samples')
    parser.add_argument('--validate_every', type=int, default=100, help='validate every n iterations')
    parser.add_argument('--sample_every', type=int, default=1000, help='sample every n iterations')
    parser.add_argument('--save_every', type=int, default=1000, help='save model every n iterations')
    parser.add_argument('--unet_number', type=int, default=1, help='unet number')
    parser.add_argument('--num_save_checkpoint', type=int, default=3, help='number of checkpoints to save')
    parser.add_argument('--exp_name', type=str, default='imagen', help='experiment name')
    args = parser.parse_args()

    if 'exp' not in os.listdir(args.save_path):
        os.mkdir(args.save_path + '/exp')

    args.save_path = args.save_path + '/exp/'

    if args.exp_name not in os.listdir(args.save_path):
        os.mkdir(args.save_path + args.exp_name)

    if f'unet{args.unet_number}' not in os.listdir(args.save_path + args.exp_name):
        os.mkdir(args.save_path + args.exp_name + f'/unet{args.unet_number}')
        os.mkdir(args.save_path + args.exp_name + f'/unet{args.unet_number}' + '/checkpoints')
        os.mkdir(args.save_path + args.exp_name + f'/unet{args.unet_number}' + '/samples')

    args.model_save_path = args.save_path + args.exp_name + f'/unet{args.unet_number}' + '/checkpoints/'
    args.sample_save_path = args.save_path + args.exp_name + f'/unet{args.unet_number}' + '/samples/'

    # unet for imagen

    unet1 = Unet(
        dim = 32,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
        layer_cross_attns = (False, True, True, True)
    )

    unet2 = Unet(
        dim = 32,
        cond_dim = 512,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True)
    )

    # imagen, which contains the unets above (base unet and super resoluting ones)

    imagen = Imagen(
        condition_on_text = True,  # this must be set to False for unconditional Imagen
        unets = (unet1, unet2),
        image_sizes = (64, 128),
        timesteps = 1000,
        channels=1,
        cond_drop_prob = 0.1
    )

    trainer = ImagenTrainer(
        imagen = imagen,
        split_valid_from_train = False, # whether to split the validation dataset from the training
    ).cuda()

    if (args.load_model):
        trainer.load(args.model_path)

    # instantiate the train, validation and datasets, which return the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks.
    dataset_train = NLMCXRDataset(args.image_path, args.text_path, image_size=64, mode='train')
    dataset_val = NLMCXRDataset(args.image_path, args.text_path, image_size=64, mode='val')
    dataset_test = NLMCXRDataset(args.image_path, args.text_path, image_size=64, mode='test')

    trainer.add_train_dataset(dataset_train, batch_size = args.batch_size)
    trainer.add_valid_dataset(dataset_val, batch_size = args.batch_size)

    # working training loop

    for i in range(args.iterations):
        loss = trainer.train_step(unet_number = args.unet_number)
        # print(f'loss: {loss}')

        if not (i % args.validate_every):
            valid_loss = trainer.valid_step(unet_number = args.unet_number)
            print(f'valid loss in iter {i}: {valid_loss}')

        if not (i % args.sample_every) and trainer.is_main: # is_main makes sure this can run in distributed
            idx = random.randint(0, len(dataset_val) - 1)
            images = trainer.sample(batch_size = 1, return_pil_images = True, text_embeds=dataset_val[idx][1].unsqueeze(0), stop_at_unet_number=args.unet_number) # returns List[Image]
            images[0].save(args.sample_save_path + f'sample-{i // args.sample_every}.png')
            print(f'saved sample {i // args.sample_every}')

        if not (i % args.save_every) and trainer.is_main:
            model_name = f'imagen-unet{args.unet_number}-{i // args.save_every}.pt'
            trainer.save(args.model_save_path + model_name)
            print(f'saved model {model_name}')
            
            if i > args.save_every:
                if f'imagen-unet{args.unet_number}-{(i // args.save_every) - args.num_save_checkpoint}.pt' in os.listdir(args.model_save_path):
                    os.remove(args.model_save_path + f'imagen-unet{args.unet_number}-{(i // args.save_every) - args.num_save_checkpoint}.pt')