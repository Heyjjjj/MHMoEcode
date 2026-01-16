import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, focal_loss, Boudaryloss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score
from utils.dataloader import augmentationimage as ugmentationimage


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, loss_fuc, num_classes, save_dir, no_improve_count):
    total_loss = 0
    val_loss = 0
    val_f_score = 0

    #print('Start Train')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=1.0)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs = batch
        #print(imgs.numpy().shape,pngs.numpy().shape)
        #imgs , pngs = ugmentationimage(imgs,pngs)
        with torch.no_grad():
            imgs = imgs.to('cuda', non_blocking=True)
            pngs = pngs.to('cuda', non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model_train(imgs)  # [bsz, 24, 448, 448]
        pngs = pngs.unsqueeze(1)


        #----------------------------------
        # choose the loss fuc
        #----------------------------------
        if loss_fuc == "BCEloss":
            loss = CE_Loss(outputs, pngs)
        
        elif loss_fuc == "Diceloss":
            loss = Dice_loss(outputs, pngs)
        
        elif loss_fuc == "Focalloss":
            loss_fn = focal_loss()
            loss = loss_fn(outputs, pngs) 
        
        elif loss_fuc == "BCEDiceloss":
            loss = CE_Loss(outputs, pngs) 
            loss = Dice_loss(outputs, pngs) * 0.1 + loss
            
        elif loss_fuc == "FocalDiceloss":
            loss_fn = focal_loss()
            loss = loss_fn(outputs, pngs) 
            loss = Dice_loss(outputs, pngs) + loss
            
        elif loss_fuc == "DiceBoudaryloss":
            a = 0.01 #weight of boudaryloss
            loss = Dice_loss(outputs, pngs)
            loss = Boudaryloss(outputs, pngs) * a + loss
        #elif loss_fuc == "PolyDiceloss":
        #    loss = CE_Loss(outputs, pngs)
        #    loss = Dice_loss(outputs, pngs) + loss
        #    loss = (f.sigmid(outputs) * pngs)
        # if dice_loss:
        #     main_dice = Dice_loss(outputs, pngs)
        #     loss = loss + main_dice

        # with torch.no_grad():
        #     #-------------------------------#
        #     #   计算f_score
        #     #-------------------------------#
        #     _f_score = f_score(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # total_f_score   += _f_score.item()
        
        pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()
    
    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs = batch
        with torch.no_grad():
            imgs = imgs.to('cuda', non_blocking=True)
            pngs = pngs.to('cuda', non_blocking=True)

            outputs = model_train(imgs)
            pngs = pngs.unsqueeze(1)
            #----------------------------------
            # choose the loss fuc
            #----------------------------------
            if loss_fuc == "BCEloss":
                loss = CE_Loss(outputs, pngs)
            
            elif loss_fuc == "Diceloss":
                loss = Dice_loss(outputs, pngs)
            
            elif loss_fuc == "Focalloss":
                loss_fn = focal_loss()
                loss = loss_fn(outputs, pngs) 
            
            elif loss_fuc == "FocalDiceloss":
                loss_fn = focal_loss()
                loss = loss_fn(outputs, pngs) 
                loss = Dice_loss(outputs, pngs) + loss
            elif loss_fuc == "DiceBoudaryloss":
                a = 0.01 #weight of boudaryloss
                loss = Dice_loss(outputs, pngs)
                loss = Boudaryloss(outputs, pngs) * a + loss

            val_loss += loss.item()
            # val_f_score += _f_score.item()

        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1), 'f_score': val_f_score / (iteration + 1), 'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()

    print('Finish Validation')
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
    # eval_callback.on_epoch_end(epoch + 1, model_train)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

    # -----------------------------------------------#
    #   保存权值
    # -----------------------------------------------#
    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
    no_improve_count += 1
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        no_improve_count = 0

    return no_improve_count