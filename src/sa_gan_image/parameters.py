import argparse
import datetime
import os


def get_parameters():

    parser = argparse.ArgumentParser()

    # Images data path & Output path
    parser.add_argument('--dataset', type=str, default='folder', choices=["cifar10", "fake", "folder", "hdf5", "imagenet", "lfw", "lsun"],
                        help="cifar10 | fake | folder | hdf5 | imagenet | lfw | lsun")
    parser.add_argument('--data_path', type=str, default='', help='Path to root of image data (saved in dirs of classes)')
    parser.add_argument('--save_path', type=str, default='./sagan_models')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_in_gpu', type=int, default=0,
                        help='0 => same as batch_size, else: if using multiple gpu iterations to make an effective batch, e.g. batch_size=32, batch_size_in_gpu=16 => optimizer.step() is run 2 iterations after running loss.backward()')
    parser.add_argument('--total_step', type=int, default=200000, help='how many iterations')
    parser.add_argument('--d_steps_per_iter', type=int, default=1, help='how many D updates per iteration')
    parser.add_argument('--g_steps_per_iter', type=int, default=1, help='how many G updates per iteration')
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Model hyper-parameters
    parser.add_argument('--adv_loss', type=str, default='hinge', choices=['hinge', 'dcgan', 'wgan_gp', 'gan'])
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10)

    # Instance noise
    # https://github.com/soumith/ganhacks/issues/14#issuecomment-312509518
    # https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    parser.add_argument('--inst_noise_sigma', type=float, default=0.0)
    parser.add_argument('--inst_noise_sigma_iters', type=int, default=2000)

    # Image transforms
    parser.add_argument('--dont_shuffle', action='store_true')
    parser.add_argument('--dont_drop_last', action='store_true', help="Whether not to drop the last batch in dataset if its size < batch_size")
    parser.add_argument('--dont_resize', action='store_true', help="Whether not to resize images")
    parser.add_argument('--imsize', type=int, default=128)
    parser.add_argument('--centercrop', action='store_true', help="Whether to center crop images")
    parser.add_argument('--centercrop_size', type=int, default=128)
    parser.add_argument('--dont_normalize', action='store_true', help="Whether to normalize image values")

    # Step sizes
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=float, default=50)
    parser.add_argument('--save_n_images', type=int, default=0,
                        help='0 => same as batch_size_in_gpu')
    parser.add_argument('--nrow', type=int, default=10)
    parser.add_argument('--max_frames_per_gif', type=int, default=100)

    # Pretrained model
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--state_dict_or_model', type=str, default='', help="Specify whether .pth pretrained_model is a 'state_dict' or a complete 'model'")

    # Misc
    parser.add_argument('--manual_seed', type=int, default=29)
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--parallel', action='store_true', help="Run on multiple GPUs")
    parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('--use_tensorboard', action='store_true')

    # Output paths
    parser.add_argument('--model_weights_dir', type=str, default='weights')
    parser.add_argument('--sample_images_dir', type=str, default='samples')

    # Model name
    parser.add_argument('--name', type=str, default='sagan')

    args = parser.parse_args()

    if args.batch_size_in_gpu == 0:
        args.batch_size_in_gpu = args.batch_size

    # asserts are to be avoided in Python because they are cleared in production code
    assert args.batch_size_in_gpu <= args.batch_size, "ERROR: please make sure batch_size >= batch_size_in_gpu!! Given batch_size: " + str(args.batch_size) + " ; batch_size_in_gpu: " + str(args.batch_size_in_gpu)
    assert args.batch_size % args.batch_size_in_gpu == 0, "ERROR: please make sure batch_size_in_gpu divides batch_size!! Given batch_size: " + str(args.batch_size) + " ; batch_size_in_gpu: " + str(args.batch_size_in_gpu)

    args.batch_size_effective = args.batch_size_in_gpu*(args.batch_size//args.batch_size_in_gpu)

    print("Effective BATCH SIZE:", args.batch_size_effective)

    if args.save_n_images == 0:
        args.save_n_images = args.batch_size_in_gpu

    # asserts are to be avoided in Python because they are cleared in production code
    assert args.save_n_images <= args.batch_size_in_gpu, "ERROR: please make save_n_images <= batch_size_in_gpu!! Given save_n_images: " + str(args.save_n_images) + " ; batch_size_in_gpu: " + str(args.batch_size_in_gpu)

    # Corrections
    args.shuffle = not args.dont_shuffle
    args.drop_last = not args.dont_drop_last
    args.resize = not args.dont_resize
    args.normalize = not args.dont_normalize

    args.dataloader_args = {'num_workers': args.num_workers}

    args.name = '{0:%Y%m%d_%H%M%S}_{1}_{2}'.format(datetime.datetime.now(), args.name, os.path.basename(args.data_path))

    args.save_path = os.path.join(args.save_path, args.name)
    args.model_weights_path = os.path.join(args.save_path, args.model_weights_dir)
    args.sample_images_path = os.path.join(args.save_path, args.sample_images_dir)

    return args
