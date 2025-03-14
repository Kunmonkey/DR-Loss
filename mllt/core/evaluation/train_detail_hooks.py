from mmcv.runner.hooks import Hook,HOOKS
import torch
from collections import OrderedDict
import mmcv
import numpy as np

@HOOKS.register_module()
class TrainDetailHook(Hook):

    def __init__(self,iter_interval=10,split='./pkl'):
        self.iter_interval = iter_interval
        self.split = mmcv.load(split)
    def after_train_iter(self,runner):
        # print('current iter is {}'.format(runner.iter),runner.log_buffer.output)
        # prob_res = self.pos_neg_res(runner)
        # # grad_res = runner.outputs['grad']
        # classifier_res = self.grad_norm(runner)
        #### 1-2
        # for k,v in prob_res.items():
        #     runner.log_buffer.output[k] = v
        # # for k,v in grad_res.items():
        # #     runner.log_buffer.output[k] = v
        # for k,v in classifier_res.items():
        #     runner.log_buffer.output[k] = v
        # runner.log_buffer.output(prob_res,1)
        if self.every_n_iters(runner,self.iter_interval):
            # print('current iter is {}'.format(runner.iter))
            runner.log_buffer.ready = True

    def after_train_epoch(self,runner):
        ### 1-3
        # grad_res = self.grad_norm(runner)
        # for k,v in grad_res.items():
        #     runner.log_buffer.output[k] = v
        # runner.log_buffer.update(grad_res,1)
        runner.log_buffer.ready = True

    def pos_neg_res(self,runner):
        # record the predicted probability of postive and negtive labels respectively 
        # on different classes, three class split, overall classes.
        # num_samples = runner.outputs['num_samples']
        prob_res = OrderedDict()
        pred_prob = torch.sigmoid(runner.outputs['pred_logit']).cpu().detach().numpy()
        # loss_all = runner.outputs['loss_all'].cpu().detach().numpy()
        # neg_scale = runner.outputs['neg_scale'].cpu().detach().numpy()
        # pos_scale = runner.outputs['pos_scale'].cpu().detach().numpy()
        labels = runner.outputs['labels'].cpu().numpy()
        num_samples = labels.shape[0]
        pos_nums = labels.sum(axis=0)
        pos_prob = labels * pred_prob
        neg_prob = (1 - labels) * (1 - pred_prob)
        # print('num_samples:{}'.format(num_samples))
        avg_cls_pos_prob = pos_prob.sum(axis=0) / pos_nums
        # avg_cls_pos_prob[pos_nums==0] = 0
        prob_res['avg_cls_pos_prob'] = avg_cls_pos_prob
        prob_res['avg_cls_neg_prob'] = neg_prob.sum(axis=0) / (num_samples - pos_nums)
        prob_res['avg_tot_pos_prob'] = pos_prob.sum() / pos_nums.sum()
        prob_res['avg_tot_neg_prob'] = neg_prob.sum() / (num_samples - pos_nums).sum()
        for k,v in self.split.items():
             if labels[:,list(v)].sum() == 0:
                continue
             pos_split_prob = np.nanmean(prob_res['avg_cls_pos_prob'][list(v)])
             neg_split_prob = np.nanmean(prob_res['avg_cls_neg_prob'][list(v)])
             prob_res[k + '_nVp'] = neg_split_prob / pos_split_prob

        ######## relative acc
        # pp = pred_prob * labels
        # p_sort = np.sort(pred_prob,axis=1)
        # p_ind = np.argsort(pred_prob)
        # num = labels.sum(axis=1)
        # corr_num = 0
        # for i in range(len(num)):
        #     corr_num += len(np.where(pp[i]>p_sort[i,-num[i]])[0])
        # rel_acc = corr_num / num.sum()
        # prob_res['rel_acc'] = rel_acc

        return prob_res

    def grad_norm(self,runner,part=['module.head.fc_cls']):
        # record the gradient and l2 norm of the gradient of different classes' classfier weight
        grad_res = OrderedDict()
        # print(runner.model.state_dict())
        params = []
        for name,param in runner.model.named_parameters():
            for i in part:
                if name == i+'.weight':
                    params.append(param)
        # params = [runner.model.state_dict()[i + '.weight'] for i in part]
        assert len(params) > 0
        grads = []
        for param in params:
            # print(param.requires_grad)
            if not param.requires_grad:
                raise Exception('this param has no grad')
            grads.append(param.grad)
        grads_norm = [torch.norm(i,p=2,dim=1).cpu().numpy() for i in grads]
        cls_norm = [torch.norm(i,p=2,dim=1).cpu().detach().numpy() for i in params]
        grad_res['classifier_grad_norm'] = grads_norm
        grad_res['classifier_norm'] = cls_norm
        return grad_res

    # def logtis_grad(self,runner):
    #     grad_res = OrderedDict()
    #     pred_logits = runner.outputs['pred_logit']
    #     loss = runner.outputs['loss']
    #     grad = torch.autograd.grad(outputs=loss,inputs=pred_logits).cpu().detach().numpy()
    #     labels = runner.outputs['labels'].cpu().numpy()
    #     num_samples = labels.shape[0]
    #     pos_nums = labels.sum(axis=0)
    #     pos_grad = grad * labels
    #     neg_grad = grad * (1 - labels)
    #     avg_cls_pos_grad = pos_grad.sum(axis=0) / pos_nums
    #     # avg_cls_pos_prob[pos_nums==0] = 0
    #     grad_res['avg_cls_pos_grad'] = avg_cls_pos_grad
    #     grad_res['avg_cls_neg_grad'] = neg_grad.sum(axis=0) / (num_samples - pos_nums)
    #     grad_res['avg_tot_pos_grad'] = pos_grad.sum() / pos_nums.sum()
    #     grad_res['avg_tot_neg_grad'] = neg_grad.sum() / (num_samples - pos_nums).sum()
    #     for k,v in self.split.items():
    #          if labels[:,list(v)].sum() == 0:
    #             continue
    #          grad_res[k + '_pos_grad']  = np.nanmean(grad_res['avg_cls_pos_grad'][list(v)])
    #          grad_res[k + '_neg_grad']  = np.nanmean(grad_res['avg_cls_neg_grad'][list(v)])

    #     return grad_res


    def vis_final_feat(self,runner):
        # visualize the feature map after dimension reduction of the final model
        pass