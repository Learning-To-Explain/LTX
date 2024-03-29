import os
from main.seg_classification.model_types_loading import CONVNET_MODELS_BY_NAME, \
    load_explainer_explaniee_models_and_feature_extractor
from pathlib import Path
from icecream import ic
import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from numpy import *
from tqdm import tqdm
from main.seg_classification.image_token_data_module_opt_segmentation import ImageSegOptDataModuleSegmentation
from config import config
from main.segmentation_eval.segmentation_dataset import get_segmentation_dataset
from utils.iou import IoU
from main.segmentation_eval.imagenet import ImagenetSegmentation, Imagenet_Segmentation_Loop
from main.segmentation_eval.segmentation_model_opt import \
    OptImageClassificationWithTokenClassificationModelSegmentation
from utils.vit_utils import (get_warmup_steps_and_total_training_steps,
                             get_loss_multipliers,
                             freeze_multitask_model,
                             get_params_from_config,
                             suppress_warnings,
                             get_backbone_details,
                             )
from utils.consts import (
    IMAGENET_VAL_IMAGES_FOLDER_PATH,
    EXPERIMENTS_FOLDER_PATH,
    MODEL_ALIAS_MAPPING,
    MODEL_OPTIONS,
    SEGMENTATION_DATASET_OPTIONS,
)
from main.segmentation_eval.segmentation_utils import (print_segmentation_results,
                                                       init_get_normalize_and_transform,
                                                       eval_results_per_res,
                                                       )
import pytorch_lightning as pl
import gc
from PIL import ImageFile
import argparse
from distutils.util import strtobool

seed_everything(config["general"]["seed"])
suppress_warnings()
ImageFile.LOAD_TRUNCATED_IMAGES = True
gc.collect()
num_workers = 0

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

if __name__ == '__main__':
    """
    # VOC
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/segmentation_eval/seg_stage_b.py --optimize-by-pos True --explainer-model-name densenet --explainee-model-name densenet --dataset-type voc &> nohups_logs/journal/eval/train_6000/seg_stage_b_densenet_densenet_opt_by_pos_voc.out &
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./:$PYTHONPATH nohup python main/segmentation_eval/seg_stage_b.py --optimize-by-pos False --explainer-model-name densenet --explainee-model-name densenet --dataset-type voc &> nohups_logs/journal/eval/train_6000/seg_stage_b_densenet_densenet_opt_by_neg_voc.out &
    
    # COCO
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./:$PYTHONPATH nohup python main/segmentation_eval/seg_stage_b.py --optimize-by-pos True --explainer-model-name densenet --explainee-model-name densenet --dataset-type coco &> nohups_logs/journal/eval/train_6000/seg_stage_b_densenet_densenet_opt_by_pos_coco.out &
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./:$PYTHONPATH nohup python main/segmentation_eval/seg_stage_b.py --optimize-by-pos False --explainer-model-name densenet --explainee-model-name densenet --dataset-type coco &> nohups_logs/journal/eval/train_6000/seg_stage_b_densenet_densenet_opt_by_neg_coco.out &
    
    """
    params_config = get_params_from_config(config_vit=config["vit"])
    parser = argparse.ArgumentParser(description='Run segmentation of LTX model')
    parser.add_argument('--explainer-model-name', type=str, default="densenet", choices=MODEL_OPTIONS)
    parser.add_argument('--explainee-model-name', type=str, default="densenet", choices=MODEL_OPTIONS)
    parser.add_argument('--dataset-type', type=str, default="imagenet", choices=SEGMENTATION_DATASET_OPTIONS)
    parser.add_argument("--optimize-by-pos",
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=params_config["optimize_by_pos"],
                        )

    parser.add_argument("--verbose",
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=params_config["verbose"],
                        )

    parser.add_argument('--n_epochs_to_optimize_stage_b',
                        type=int,
                        default=params_config["n_epochs_to_optimize_stage_b"],
                        )
    parser.add_argument('--n-epochs', type=int, default=params_config["n_epochs"])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--is-freezing-explaniee-model',
                        type=lambda x: bool(strtobool(x)),
                        nargs="?",
                        const=True,
                        default=params_config["is_freezing_explaniee_model"])
    parser.add_argument('--explainer-model-n-first-layers-to-freeze',
                        type=int,
                        default=params_config["explainer_model_n_first_layers_to_freeze"])
    parser.add_argument('--is-clamp-between-0-to-1',
                        type=lambda x: bool(strtobool(x)),
                        nargs="?",
                        const=True,
                        default=params_config["is_clamp_between_0_to_1"])
    parser.add_argument('--is-competitive-method-transforms',
                        type=lambda x: bool(strtobool(x)),
                        nargs="?",
                        const=True,
                        default=params_config["is_competitive_method_transforms"])
    parser.add_argument('--plot-path', type=str, default=params_config["plot_path"])
    parser.add_argument('--default-root-dir', type=str, default=params_config["default_root_dir"])
    parser.add_argument('--prediction-loss-mul', type=int, default=params_config["prediction_loss_mul"])
    parser.add_argument('--mask-loss', type=str, default=params_config["mask_loss"])
    parser.add_argument('--train-n-label-sample', type=str, default=params_config["train_n_label_sample"])
    parser.add_argument('--lr', type=float, default=params_config["lr"])
    parser.add_argument('--start-epoch-to-evaluate', type=int, default=params_config["start_epoch_to_evaluate"])
    parser.add_argument('--n-batches-to-visualize', type=int, default=params_config["n_batches_to_visualize"])
    parser.add_argument('--is-ce-neg', type=str, default=params_config["is_ce_neg"])
    parser.add_argument('--activation-function', type=str, default=params_config["activation_function"])
    parser.add_argument('--use-logits-only',
                        type=lambda x: bool(strtobool(x)),
                        nargs="?",
                        const=True,
                        default=params_config["use_logits_only"],
                        )
    parser.add_argument('--evaluation-experiment-folder-name',
                        type=str,
                        default=params_config["evaluation_experiment_folder_name"],
                        )

    args = parser.parse_args()

    EXPLAINEE_MODEL_NAME, EXPLAINER_MODEL_NAME = MODEL_ALIAS_MAPPING[args.explainee_model_name], \
                                                 MODEL_ALIAS_MAPPING[args.explainer_model_name]

    IS_EXPLANIEE_CONVNET = True if EXPLAINEE_MODEL_NAME in CONVNET_MODELS_BY_NAME.keys() else False
    IS_EXPLAINER_CONVNET = True if EXPLAINER_MODEL_NAME in CONVNET_MODELS_BY_NAME.keys() else False

    CKPT_PATH, IMG_SIZE, PATCH_SIZE, MASK_LOSS_MUL, CHECKPOINT_EPOCH_IDX, BASE_CKPT_MODEL_AUC = get_backbone_details(
        explainer_model_name=args.explainer_model_name,
        explainee_model_name=args.explainee_model_name,
        target_or_predicted_model="predicted",
    )
    loss_multipliers = get_loss_multipliers(normalize=False,
                                            mask_loss_mul=MASK_LOSS_MUL,
                                            prediction_loss_mul=args.prediction_loss_mul,
                                            )
    ic(CKPT_PATH)
    ic(MASK_LOSS_MUL)
    ic(args.explainer_model_n_first_layers_to_freeze)
    ic(args.n_epochs_to_optimize_stage_b)
    ic(args.use_logits_only)
    ic(args.dataset_type)

    test_img_trans, test_img_trans_only_resize, test_lbl_trans = init_get_normalize_and_transform()
    ds = get_segmentation_dataset(dataset_type=args.dataset_type,
                                  batch_size=args.batch_size,
                                  test_img_trans=test_img_trans,
                                  test_img_trans_only_resize=test_img_trans_only_resize,
                                  test_lbl_trans=test_lbl_trans,
                                  )

    BASE_AUC_OBJECTS_PATH = Path(EXPERIMENTS_FOLDER_PATH, args.evaluation_experiment_folder_name)
    model_for_classification_image, model_for_mask_generation, feature_extractor = load_explainer_explaniee_models_and_feature_extractor(
        explainee_model_name=EXPLAINEE_MODEL_NAME,
        explainer_model_name=EXPLAINER_MODEL_NAME,
        activation_function=args.activation_function,
        img_size=IMG_SIZE,
    )

    warmup_steps, total_training_steps = get_warmup_steps_and_total_training_steps(
        n_epochs=args.n_epochs_to_optimize_stage_b,
        train_samples_length=len(list(Path(IMAGENET_VAL_IMAGES_FOLDER_PATH).iterdir())),
        batch_size=args.batch_size,
    )

    metric = IoU(num_classes=2, ignore_index=-1)

    model = OptImageClassificationWithTokenClassificationModelSegmentation(
        model_for_classification_image=model_for_classification_image,
        model_for_mask_generation=model_for_mask_generation,
        plot_path='',
        warmup_steps=warmup_steps,
        total_training_steps=total_training_steps,
        best_auc_objects_path=Path(""),
        checkpoint_epoch_idx=CHECKPOINT_EPOCH_IDX,
        best_auc_plot_path='',
        run_base_model_only=False,
        model_runtype='train',
        experiment_path='exp_name',
        is_explainer_convnet=IS_EXPLAINER_CONVNET,
        is_explainee_convnet=IS_EXPLANIEE_CONVNET,
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        start_epoch_to_evaluate=args.start_epoch_to_evaluate,
        n_batches_to_visualize=args.n_batches_to_visualize,
        mask_loss=args.mask_loss,
        mask_loss_mul=MASK_LOSS_MUL,
        prediction_loss_mul=args.prediction_loss_mul,
        activation_function=args.activation_function,
        train_model_by_target_gt_class=False,
        use_logits_only=args.use_logits_only,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        is_ce_neg=args.is_ce_neg,
        verbose=args.verbose,
        optimize_by_pos=args.optimize_by_pos,
    )

    model = freeze_multitask_model(
        model=model,
        is_freezing_explaniee_model=args.is_freezing_explaniee_model,
        explainer_model_n_first_layers_to_freeze=args.explainer_model_n_first_layers_to_freeze,
        is_explainer_convnet=IS_EXPLAINER_CONVNET,

    )
    total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    total_ap, total_f1 = [], []
    predictions, targets = [], []
    random.seed(config["general"]["seed"])
    n_batches = range(len(ds))
    for batch_idx in tqdm(n_batches, leave=True, position=0):
        ds_loop = Imagenet_Segmentation_Loop(*ds[batch_idx])
        dl = DataLoader(ds_loop, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)
        data_module = ImageSegOptDataModuleSegmentation(
            train_data_loader=dl
        )
        trainer = pl.Trainer(
            logger=[],
            accelerator='gpu',
            gpus=1,
            devices=[1, 2],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=300,
            max_epochs=CHECKPOINT_EPOCH_IDX + args.n_epochs_to_optimize_stage_b,
            resume_from_checkpoint=CKPT_PATH,
            enable_progress_bar=False,
            enable_checkpointing=False,
            default_root_dir=args.default_root_dir,
            weights_summary=None
        )

        trainer.fit(model=model, datamodule=data_module)

        image_resized = model.image_resized
        Res = model.best_auc_vis
        labels = model.target
        correct, labeled, inter, union, ap, f1, pred, target = eval_results_per_res(Res=Res,
                                                                                    labels=labels,
                                                                                    index=batch_idx,
                                                                                    image=image_resized)
        predictions.append(pred)
        targets.append(target)

        total_correct += correct.astype('int64')
        total_label += labeled.astype('int64')
        total_inter += inter.astype('int64')
        total_union += union.astype('int64')
        total_ap += [ap]
        total_f1 += [f1]
        pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
        IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
        mIoU = IoU.mean()
        mAp = np.mean(total_ap)
        mF1 = np.mean(total_f1)
        if (batch_idx % 100 == 0) or (batch_idx == n_batches[-1]):
            print_segmentation_results(pixAcc=pixAcc, mAp=mAp, mIoU=mIoU, mF1=mF1)

    print_segmentation_results(pixAcc=pixAcc, mAp=mAp, mIoU=mIoU, mF1=mF1)
