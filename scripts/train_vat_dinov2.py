import argparse
from datetime import datetime
import numpy as np
import os
import random
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import wandb

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.model_dinov2 import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2
from visualize import plot_gazelle_results

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gazelle_dinov2_vitb14_inout")
parser.add_argument('--init_ckpt', type=str, default='./checkpoints/gazelle_dinov3_vitb16.pt', help='checkpoint for initialization (trained on GazeFollow)')
parser.add_argument('--data_path', type=str, default='/newhome/fb/dataset/videoattentiontarget')
parser.add_argument('--frame_sample_every', type=int, default=6)
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--imgs_save_dir', type=str, default='./experiments_imgs')
parser.add_argument('--wandb_project', type=str, default='gazelleV1')
parser.add_argument('--exp_name', type=str, default='train_vat')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--inout_loss_lambda', type=float, default=1.0)
parser.add_argument('--lr_non_inout', type=float, default=1e-5)
parser.add_argument('--lr_inout', type=float, default=1e-3)
parser.add_argument('--n_workers', type=int, default=8)
# 【新增】 Contribution 开关参数
parser.add_argument('--use_sasa', action='store_true', help='Enable Scale-Aware Semantic Aggregation (Contribution 1)')
parser.add_argument('--use_ggsf', action='store_true', help='Enable Geometry-Guided Spatial Focus (Contribution 2)')
parser.add_argument('--use_aux', action='store_true', help='Enable Auxiliary Loss (Contribution 3)')
parser.add_argument('--aux_weight', type=float, default=0.3, help='Weight for the auxiliary loss')
args = parser.parse_args()


def main():
    wandb.init(
        project=args.wandb_project,
        name=args.exp_name,
        config=vars(args)
    )
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(exp_dir)

    imgs_dir = os.path.join(args.imgs_save_dir, args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(imgs_dir)

    print(f"Experimental Config - SASA: {args.use_sasa}, GGSF: {args.use_ggsf}, AUX: {args.use_aux}")


    model, transform = get_gazelle_model(
        args.model, 
        use_sasa=args.use_sasa, 
        use_ggsf=args.use_ggsf, 
        use_aux=args.use_aux
    )
    print("Initializing from {}".format(args.init_ckpt))
    model.load_gazelle_state_dict(torch.load(args.init_ckpt, weights_only=True)) # initializing from ckpt without inout head
    model.cuda()

    for param in model.backbone.parameters(): # freeze backbone
        param.requires_grad = False
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_dataset = GazeDataset('videoattentiontarget', args.data_path, 'train', transform, in_frame_only=False, sample_rate=args.frame_sample_every)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    # Note this eval dataloader samples frames sparsely for efficiency - for final results, run eval_vat.py which uses sample rate 1
    eval_dataset = GazeDataset('videoattentiontarget', args.data_path, 'test', transform, in_frame_only=False, sample_rate=args.frame_sample_every)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    loss_fn = nn.BCELoss()
    inout_loss_fn = nn.BCELoss()
    param_groups = [
        {'params': [param for name, param in model.named_parameters() if "inout" in name], 'lr': args.lr_inout},
        {'params': [param for name, param in model.named_parameters() if "inout" not in name], 'lr': args.lr_non_inout}
    ]
    optimizer = torch.optim.Adam(param_groups)

    for epoch in range(args.max_epochs):
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]})
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)

            # compute heatmap loss only for in-frame gaze targets
            heatmap_loss = loss_fn(heatmap_preds[inout.bool()], heatmaps[inout.bool()].cuda())
            inout_loss = inout_loss_fn(inout_preds, inout.float().cuda())
            
            loss_aux = torch.tensor(0.0).cuda()
            if args.use_aux and preds.get("aux_heatmap") is not None:
                # 注意：preds['aux_heatmap'] 也是一个 list，需要 stack
                aux_preds_stack = torch.stack(preds['aux_heatmap']).squeeze(dim=1)
                loss_aux = loss_fn(aux_preds_stack[inout.bool()], heatmaps[inout.bool()].cuda())
                
                # 总 Loss = Main + Weight * Aux
                loss = heatmap_loss + args.aux_weight * loss_aux + args.inout_loss_lambda * inout_loss
            else:
                loss = heatmap_loss + args.inout_loss_lambda * inout_loss

            loss.backward()
            optimizer.step()

            if cur_iter % args.log_iter == 0:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/heatmap_loss": heatmap_loss.item(),
                    "train/inout_loss": inout_loss.item()
                }
                if args.use_aux:
                    log_dict["train/loss_aux"] = loss_aux.item()
                wandb.log(log_dict)
                log_msg = "TRAIN EPOCH {}, iter {}/{}, loss={}".format(epoch, cur_iter, len(train_dl), round(loss.item(), 4))
                if args.use_aux:
                    log_msg += " (Main: {:.4f}, Aux: {:.4f})".format(heatmap_loss.item(), loss_aux.item())
                print(log_msg)

                # img_path = os.path.join(imgs_dir, f"vis_epoch_{epoch}_batch_{cur_iter}.png")
                # # 可视化 Batch 中的第一张图片 (index=0)
                # plot_gazelle_results(
                #     input_data={"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]},
                #     model_output=preds,
                #     index=0,
                #     save_path=img_path
                # )

        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_gazelle_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

        # EVAL EPOCH
        print("Running evaluation")
        model.eval()
        l2s = []
        aucs = []
        all_inout_preds = []
        all_inout_gts = []
        for cur_iter, batch in enumerate(eval_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            with torch.no_grad():
                preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
            for i in range(heatmap_preds.shape[0]):
                if inout[i] == 1: # in-frame
                    auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    aucs.append(auc)
                    l2s.append(l2)
                all_inout_preds.append(inout_preds[i].item())
                all_inout_gts.append(inout[i])

        epoch_l2 = np.mean(l2s)
        epoch_auc = np.mean(aucs)
        epoch_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)

        wandb.log({"eval/auc": epoch_auc, "eval/l2": epoch_l2, "eval/inout_ap": epoch_inout_ap, "epoch": epoch})
        print("EVAL EPOCH {}: AUC={}, L2={}, Inout AP={}".format(epoch, round(epoch_auc, 4), round(epoch_l2, 4), round(epoch_inout_ap, 4)))


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()