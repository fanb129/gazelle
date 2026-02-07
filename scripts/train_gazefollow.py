import argparse
from datetime import datetime
import numpy as np
import os
import random
import torch
import torch.nn as nn
import wandb

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gazelle_dinov3_vitb16")
parser.add_argument('--data_path', type=str, default='/newhome/fb/dataset/gazefollow_extended')
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--wandb_project', type=str, default='gazelleV1')
parser.add_argument('--exp_name', type=str, default='train_gazefollow_vitb_coarse_to_fine')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
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

    print(f"Experimental Config - SASA: {args.use_sasa}, GGSF: {args.use_ggsf}, AUX: {args.use_aux}")

    # 【修改】将开关参数传递给模型
    model, transform = get_gazelle_model(
        args.model, 
        use_sasa=args.use_sasa, 
        use_ggsf=args.use_ggsf, 
        use_aux=args.use_aux
    )
    model.cuda()

    for param in model.backbone.parameters(): # freeze backbone
        param.requires_grad = False
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_dataset = GazeDataset('gazefollow', args.data_path, 'train', transform)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    best_min_l2 = 1.0
    best_epoch = None

    for epoch in range(args.max_epochs):
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]})
            
            # 主 Loss 计算
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            loss_main = loss_fn(heatmap_preds, heatmaps.cuda())
            
            # 【新增】 Aux Loss 计算 logic
            loss_aux = torch.tensor(0.0).cuda()
            if args.use_aux and preds.get("aux_heatmap") is not None:
                # 注意：preds['aux_heatmap'] 也是一个 list，需要 stack
                aux_preds_stack = torch.stack(preds['aux_heatmap']).squeeze(dim=1)
                loss_aux = loss_fn(aux_preds_stack, heatmaps.cuda())
                
                # 总 Loss = Main + Weight * Aux
                loss = loss_main + args.aux_weight * loss_aux
            else:
                loss = loss_main

            loss.backward()
            optimizer.step()

            if cur_iter % args.log_iter == 0:
                # 【修改】分别记录主Loss和辅助Loss
                log_dict = {"train/loss": loss.item(), "train/loss_main": loss_main.item()}
                if args.use_aux:
                    log_dict["train/loss_aux"] = loss_aux.item()
                
                wandb.log(log_dict)
                
                # 打印信息优化
                log_msg = "TRAIN EPOCH {}, iter {}/{}, loss={:.4f}".format(epoch, cur_iter, len(train_dl), loss.item())
                if args.use_aux:
                    log_msg += " (Main: {:.4f}, Aux: {:.4f})".format(loss_main.item(), loss_aux.item())
                print(log_msg)

        scheduler.step()

        # 保存模型时不需要特殊修改，因为 get_gazelle_state_dict 会处理好
        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_gazelle_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

        # EVAL EPOCH (Evaluation 通常不需要计算 Aux Loss，只看最终输出)
        print("Running evaluation")
        model.eval()
        avg_l2s = []
        min_l2s = []
        aucs = []
        for cur_iter, batch in enumerate(eval_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            with torch.no_grad():
                preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            for i in range(heatmap_preds.shape[0]):
                auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)

        epoch_avg_l2 = np.mean(avg_l2s)
        epoch_min_l2 = np.mean(min_l2s)
        epoch_auc = np.mean(aucs)

        wandb.log({"eval/auc": epoch_auc, "eval/min_l2": epoch_min_l2, "eval/avg_l2": epoch_avg_l2, "epoch": epoch})
        print("EVAL EPOCH {}: AUC={}, Min L2={}, Avg L2={}".format(epoch, round(epoch_auc, 4), round(epoch_min_l2, 4), round(epoch_avg_l2, 4)))

        if epoch_min_l2 < best_min_l2:
            best_min_l2 = epoch_min_l2
            best_epoch = epoch

    print("Completed training. Best Min L2 of {} obtained at epoch {}".format(round(best_min_l2, 4), best_epoch))

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()