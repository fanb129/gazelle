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
parser.add_argument('--exp_name', type=str, default='train_gazefollow_vitb')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--wandb', type=str, default='1')
args = parser.parse_args()


def main():
    # 如果 args.wandb 为 '1' 则在线运行，否则禁用
    mode = 'online' if args.wandb == '1' else 'disabled'
    debug_save_path='/home/fb/src/paper/gazelleV1/visImg' if args.wandb == '0' else None
    wandb.init(
        project=args.wandb_project,
        name=args.exp_name,
        config=vars(args),
        mode=mode  # 设置为 'disabled' 后，后续所有的 wandb.log 都会被静默忽略
    )
    
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(exp_dir)

    model, transform = get_gazelle_model(args.model)
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
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps, text_prompt = batch

            optimizer.zero_grad()
            preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "text": text_prompt}, debug_save_path)
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            # heatmap_preds = preds['heatmap']
            loss_grounding = preds["grounding_loss"]
            loss_gaze = loss_fn(heatmap_preds, heatmaps.cuda())
            total_loss = loss_gaze + 1.0 * loss_grounding
            total_loss.backward()
            optimizer.step()

            if cur_iter % args.log_iter == 0:
                wandb.log({"train/totalLoss": total_loss.item(), "train/loss_grounding": loss_grounding.item(), "train/loss_gaze": loss_gaze.item()})
                print("TRAIN EPOCH {}, iter {}/{}, totalLoss={}, loss_grounding={}, loss_gaze={}".format(epoch, cur_iter, len(train_dl), round(total_loss.item(), 4), round(loss_grounding.item(), 4), round(loss_gaze.item(), 4)))

        scheduler.step()

        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_gazelle_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

        # EVAL EPOCH
        print("Running evaluation")
        model.eval()
        avg_l2s = []
        min_l2s = []
        aucs = []
        for cur_iter, batch in enumerate(eval_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, text_prompt = batch

            with torch.no_grad():
                preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "text": text_prompt})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            # heatmap_preds = preds['heatmap']
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