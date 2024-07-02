import os
import torch
import argparse
from pathlib import Path

from src import utils
from src import head_model as head
from src import ssl_model as backbone
from src import dataloader as loader
from src import upsampler
from src.task import classification as multiclass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', default="exp1", type=str, 
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word.')
    parser.add_argument('--upsampler_conf', default=None, type=str,
                        help='Path to file with the upsampler model configurations.')
    parser.add_argument('--pretrain_conf', default=None, type=str,
                        help='Path to file with the self-supervised model configurations.')
    parser.add_argument('--task_conf', default=None, type=str,
                        help='Path to file with the downstream task model configurations.')
    parser.add_argument('--seed', default=None, type=int, help='Seed')

    args = parser.parse_args()

    if args.pretrain_conf is None or args.task_conf is None:
        assert "Either `pretrain_conf` or `task_conf` must be not None."

    # Import the parameter's files
    pretrain_conf = utils.load_json(args.pretrain_conf) if args.pretrain_conf else None
    upsampler_conf = utils.load_json(args.upsampler_conf) if args.upsampler_conf else None
    task_conf_list = utils.load_json(args.task_conf) if args.task_conf else None

    if args.seed:
        utils.set_seed(args.seed)
        # init_dl_program(device_name=0, seed=args.seed, max_threads=8)

    run_dir = Path("training") / utils.name_with_datetime(args.name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssl_model_params = None
    logs = {}

    # --------------- Call the upsampler routine --------------- #
    if upsampler_conf:
        if args.seed:
            upsampler_conf['seed'] = args.seed

        parent_dir = Path(upsampler_conf.get('parent_dir', run_dir))
        os.makedirs(parent_dir, exist_ok=True)

        # Load the pretrain dataset
        dataset_dir = 'datasets/' + upsampler_conf['dataset']
        dataset = loader.downstream_dataset(dataset_dir)

        # Pretrain the feature extractor
        upsampler_params, upsampler_logs = upsampler.pretrain(dataset,
                                                              upsampler_conf['train_params'],
                                                              upsampler_conf['model_args'], 
                                                              upsampler_conf['optim'],
                                                              device)

        # save trained ssl_model and trainning logs
        utils.save_json(parent_dir / 'upsampler_conf.json', upsampler_conf)
        utils.save_json(parent_dir / "upsampler_logs.json", upsampler_logs)
        torch.save(upsampler_params,  parent_dir / "upsampler_params.pt")


    # --------------- Call the self-supervised routine --------------- #
    if pretrain_conf:
        if args.seed:
            pretrain_conf['seed'] = args.seed

        os.makedirs(run_dir, exist_ok=True)

        print('\n', "=" * 45)
        print(f'Mode:\t Self-supervised')
        print(f'Upsampler:\t {pretrain_conf.get('upsampler_args', None) }')
        print(f'Seed:\t {args.seed}')
        print(f'Dir:\t {run_dir}')
        print("=" * 45, '\n')

        # Load the pretrain dataset
        dataset_dir = 'datasets/' + pretrain_conf['dataset']
        dataset = loader.pretrain_dataset(dataset_dir)

        # Pretrain the feature extractor
        trainer = backbone.get_trainer(pretrain_conf['model'])
        ssl_model_params, ssl_logs = trainer.pretrain(dataset, 
                                                      pretrain_conf['train_params'],
                                                      pretrain_conf['model_args'],
                                                      pretrain_conf['optim'], 
                                                      device=device)

        # save trained ssl_model and trainning logs
        utils.save_json(run_dir / 'pretrain_conf.json', pretrain_conf)
        utils.save_json(run_dir / "pretrain_logs.json", ssl_logs)
        torch.save(ssl_model_params,  run_dir / "ssl_model_params.pt")


    # --------------- Call the supervised routine --------------- #
    if task_conf_list:
        best_conf = {}
        best_metric = 0

        for i, task_conf in enumerate(task_conf_list):

            if args.seed:
                task_conf['seed'] = args.seed

            # Create running dir to save the logs, models and training params
            parent_dir = Path(task_conf['parent_dir'])
            task_dir = parent_dir / utils.task_dir(task_conf)
            temp_dir = task_dir / "temporary"
            os.makedirs(temp_dir, exist_ok=True)

            print('\n', "=" * 45)
            print(f'Mode:\t {task_conf['train_mode']}')
            print(f'Head:\t {task_conf['model']}')
            print(f'See:\t {args.seed}')
            print(f'Dir:\t {task_dir}')
            print("=" * 45, '\n')

            # Load the downstram dataset
            dataset_dir = 'datasets/' + task_conf['dataset']
            train_data, val_data, test_data = loader.downstream_dataset(dataset_dir)

            # Instance the SSL model
            ssl_model = None
            ssl_optim = None
            if task_conf['train_mode'] != 'supervised':

                # load the ssl_model config file
                if pretrain_conf is None:
                    pretrain_conf = utils.load_json(parent_dir / "pretrain_conf.json")

                # load the ssl_model pretrained parameters
                ssl_model_params = None
                if task_conf['train_mode'] != 'random':
                    ssl_model_params = torch.load(parent_dir / "ssl_model_params.pt")               
                
                ssl_model, ssl_optim = backbone.instance_model(pretrain_conf['model'], 
                                                            pretrain_conf['model_args'],
                                                            pretrain_conf['optim'], 
                                                            task_conf['train_mode'],
                                                            ssl_model_params, device)
                    
            # Instanciate the upsampler
            upsampler_model = None
            upsampler_optim = None
            if task_conf.get('upsampler', False):
                print("Has upsampler.")
                upsampler_conf = utils.load_json(parent_dir / "upsampler_conf.json")
                upsampler_params = torch.load(parent_dir / "upsampler_params.pt")

                upsampler_model = upsampler.ESPCN1d(**upsampler_conf['model_args'])
                upsampler_model.load_state_dict(upsampler_params['upsampler'])
                
                upsampler_optim = torch.optim.Adam(upsampler_model.parameters(), 
                                                    **task_conf['optim'])
                

            # Instaciate the head_model
            head_model, head_optim = head.classifier(task_conf['model'],
                                                     task_conf['model_args'], 
                                                     task_conf['optim'])
            
            # Tune the model for the downstream task
            checkpoint, sl_logs = multiclass.fit_cls(head_model, head_optim, 
                                                     (train_data, val_data), 
                                                     task_conf['train_params'], 
                                                     ssl_model, ssl_optim,
                                                     upsampler_model, upsampler_optim,
                                                     task_conf['train_mode'], 
                                                     device)

            test_logs = multiclass.evaluate_cls(head_model, test_data,
                                                task_conf['train_params'], 
                                                ssl_model, upsampler_model, device)


            if test_logs['metrics']['macro_f1'] > best_metric:
                best_metric = test_logs['metrics']['macro_f1']
                best_conf = {
                    "checkpoint": checkpoint,
                    "sl_logs": sl_logs,
                    "test_logs": test_logs,
                    "task_conf": task_conf,
                    "dir": task_dir
                }

            # Save the models' params and logs
            if len(task_conf_list) > 1:
                torch.save(checkpoint, temp_dir / f"{task_conf['id']}-checkpoint.tar")
                utils.save_json(temp_dir / f"{task_conf['id']}-sl_logs.json", sl_logs)
                utils.save_json(temp_dir / f"{task_conf['id']}-test_logs.json", test_logs)
                utils.save_json(temp_dir / f"{task_conf['id']}-task_conf.json", task_conf)
        
        # Save the models' params and logs
        torch.save(best_conf['checkpoint'], best_conf['dir'] / "checkpoint.tar")
        utils.save_json(best_conf['dir'] / "sl_logs.json", best_conf['sl_logs'])
        utils.save_json(best_conf['dir'] / "test_logs.json", best_conf['test_logs'])
        utils.save_json(best_conf['dir'] / "task_conf.json", best_conf['task_conf'])
        