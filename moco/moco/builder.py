# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

class NAC(ModelBase):
    def __init__(self, base_encoder, dim=128, p=0.1, VI=False):
        super().__init__()
        self.dim = dim
        self.p = p
        self.VI = VI
        assert VI
        self.temperature =  1 / (np.log((1 - p) / p) / 2)
        self.encoder = base_encoder(num_classes=dim)
        dim_mlp = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                  nn.ReLU(),
                                  nn.Linear(dim_mlp, dim))
        if VI:
            self.prediction = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                            nn.ReLU(),
                                            nn.Linear(dim_mlp, dim))

    def forward(self, im_1, im_2):
        h_1 =  self.encoder(im_1)
        out_1 = self.head(h_1)
        # shuffle for making use of BN
        im_2, idx_unshuffle = self._batch_shuffle_ddp(im_2)
        h_2 = self.encoder(im_2)  # keys: NxC
        # undo shuffle
        h_2 = self._batch_unshuffle_ddp(h_2, idx_unshuffle)
        out_2 = self.head(h_2)

        out_1 = torch.tanh(out_1 / np.sqrt(self.dim))
        out_2 = torch.tanh(out_2 / np.sqrt(self.dim))
        out = torch.cat([out_1, out_2], dim=0)
        # Apply symmetric noise
        m = (torch.rand_like(out) > self.p).to(torch.float32)
        m = 2 * (m - 0.5)
        out_drop = m * out

        if self.VI:
            logit_1 = self.prediction(h_1) / np.sqrt(self.dim)
            logit_2 = self.prediction(h_2) / np.sqrt(self.dim)
            logit = torch.cat([logit_1, logit_2], dim=0)
        else:
            logit = torch.zeros_like(out)

        batch_size = out_1.shape[0]
        log_sim_matrix = torch.mm(out_drop, out.t().contiguous()) / self.temperature
        q_1, q_2 = logit_1.sigmoid(), logit_2.sigmoid()
        log_pos_sim1 = (torch.sum(out_drop[:batch_size] * logit_2
                                + torch.log(q_2 * (1 - q_2)) - np.log(1/4), dim=-1)) / 2
        log_pos_sim2 = (torch.sum(logit_1 * out_drop[batch_size:]
                                + torch.log(q_1 * (1 - q_1)) - np.log(1/4), dim=-1)) / 2
        log_pos_sim = torch.cat([log_pos_sim1, log_pos_sim2], dim=0)

        m = log_sim_matrix.max(dim=-1)[0]
        loss = (- log_pos_sim
                + torch.log(torch.exp(log_sim_matrix - m.unsqueeze(1)).sum(dim=-1)) + m)
        labels = torch.arange(2 * batch_size, dtype=torch.long).cuda()

        return loss, out, log_sim_matrix, labels

class MoCoBase(ModelBase):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        # create the queue
        self.register_buffer("queue", 1e-3 * torch.randn(dim, K))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


class MoCo(MoCoBase):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__(base_encoder, dim, K, m)
        self.T = T

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.queue = nn.functional.normalize(self.queue, dim=0)

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels



class NACMoCo(MoCoBase):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, p=0.1, VI=False, shuffle_key=True, proj_bn=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__(base_encoder, dim, K, m)
        self.dim = dim
        self.p = p
        self.VI = VI
        self.shuffle_key = shuffle_key
        self.temperature =  1 / (np.log((1 - p) / p) / 2)

        if not shuffle_key:
            self.encoder_q = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
            self.encoder_k = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Identity()
        self.encoder_k.fc = nn.Identity()
        self.head_q = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                    nn.BatchNorm1d(dim_mlp) if proj_bn else nn.Identity(),
                                    nn.ReLU(),
                                    nn.Linear(dim_mlp, dim))
        self.head_k = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                    nn.BatchNorm1d(dim_mlp) if proj_bn else nn.Identity(),
                                    nn.ReLU(),
                                    nn.Linear(dim_mlp, dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if VI:
            self.prediction_q = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                              nn.BatchNorm1d(dim_mlp) if proj_bn else nn.Identity(),
                                              nn.ReLU(),
                                              nn.Linear(dim_mlp, dim))
            self.prediction_k = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                              nn.BatchNorm1d(dim_mlp) if proj_bn else nn.Identity(),
                                              nn.ReLU(),
                                              nn.Linear(dim_mlp, dim))
            for param_q, param_k in zip(self.prediction_q.parameters(), self.prediction_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        self.classifier = nn.Linear(dim_mlp, 1000)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        if self.VI:
            for param_q, param_k in zip(self.prediction_q.parameters(), self.prediction_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle, idx_this


    @torch.no_grad()
    def _batch_shuffle_ddp_given_idx(self, x, idx_this):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        x_gather = concat_all_gather(x)

        return x_gather[idx_this]

    def forward(self, im_1, im_2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        batch_size = im_1.shape[0]
        if self.shuffle_key:
            h_1 = self.encoder_q(im_1)  # queries: NxC
            out_1 = self.head_q(h_1)
            out_1 = torch.tanh(out_1 / np.sqrt(self.dim))

            # shuffle for making use of BN
            im_2_shuffled, idx_unshuffle, idx_shuffle = self._batch_shuffle_ddp(im_2)
            h_2_shuffled = self.encoder_q(im_2_shuffled)  # keys: NxC
            out_2_shuffled = self.head_q(h_2_shuffled)
            out_2_shuffled = torch.tanh(out_2_shuffled / np.sqrt(self.dim))
            # undo shuffle
            h_2 = self._batch_unshuffle_ddp(h_2_shuffled, idx_unshuffle)
            h_1_shuffled = self._batch_shuffle_ddp_given_idx(h_1, idx_shuffle)
            if self.VI:
                logit_1_shuffled = self.prediction_q(h_1_shuffled.detach()) / np.sqrt(self.dim)
                logit_2 = self.prediction_q(h_2.detach()) / np.sqrt(self.dim)
            else:
                logit_1_shuffled = torch.zeros_like(out_1)
                logit_2 = torch.zeros_like(out_1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                h_k_1 = self.encoder_k(im_1)  # keys: NxC
                # h_k_1_shuffled has the same ordering as h_2_shuffled
                h_k_1_shuffled = self._batch_shuffle_ddp_given_idx(h_k_1, idx_shuffle)
                k_1_shuffled = self.head_k(h_k_1_shuffled)
                k_1_shuffled = torch.tanh(k_1_shuffled / np.sqrt(self.dim))

                h_k_2_shuffled = self.encoder_k(im_2_shuffled)  # keys: NxC
                h_k_2 = self._batch_unshuffle_ddp(h_k_2_shuffled, idx_unshuffle)
                k_2 = self.head_k(h_k_2)
                k_2 = torch.tanh(k_2 / np.sqrt(self.dim))

                if self.VI:
                    logit_k_1_shuffled = self.prediction_k(h_k_1_shuffled) / np.sqrt(self.dim)
                    logit_k_2 = self.prediction_k(h_k_2) / np.sqrt(self.dim)
                else:
                    logit_k_1_shuffled = torch.zeros_like(out_1)
                    logit_k_2 = torch.zeros_like(out_1)

            k = torch.cat([k_2, k_1_shuffled], dim=0)
            logit = torch.cat([logit_2, logit_1_shuffled], dim=0)
            logit_k = torch.cat([logit_k_2, logit_k_1_shuffled], dim=0)
            keys = self.queue.clone().detach()
            out = torch.cat([out_1, out_2_shuffled], dim=0)

            # Apply symmetric noise
            m = (torch.rand_like(out) > self.p).to(torch.float32)
            m = 2 * (m - 0.5)
            out_drop = m * out

            l_pos = torch.einsum('nc,nc->n', [out_drop, k]).unsqueeze(-1)
            log_sim_matrix = torch.mm(out_drop, keys)
            log_sim_matrix = torch.cat([l_pos, log_sim_matrix], dim=1) / self.temperature
            m = log_sim_matrix.max(dim=-1)[0]
            if self.VI:
                q_k = logit_k.sigmoid()
                log_pos_sim = (torch.sum(out_drop * logit_k
                                         + torch.log(q_k * (1 - q_k)) - np.log(1/4), dim=-1)) / 2
                loss = (- log_pos_sim
                        + torch.logsumexp(log_sim_matrix, dim=-1))

                q = logit.sigmoid()
                KL = (torch.sum(out_drop.detach() * logit
                                + torch.log(q * (1 - q)) - np.log(1/4), dim=-1)) / 2
                loss = loss - KL
            else:
                log_pos_sim = l_pos
                loss = (- log_pos_sim / self.temperature
                        + torch.log(torch.exp(log_sim_matrix - m.unsqueeze(1)).sum(dim=-1)) + m)

            labels = torch.zeros(batch_size, dtype=torch.long).cuda()
            self._dequeue_and_enqueue(k)

            classifier_output = self.classifier(h_1.detach())

            return loss, out_1, log_sim_matrix, labels, keys, logit_k_2, logit_2, l_pos, k_2, classifier_output
        else:
            # Sync batch norm
            im = torch.cat([im_1, im_2], dim=0)
            h = self.encoder_q(im)
            h_1, h_2 = h[:batch_size], h[batch_size:]
            out = self.head_q(h)
            out = torch.tanh(out / np.sqrt(self.dim))

            if self.VI:
                logit = self.prediction_q(h.detach()) / np.sqrt(self.dim)
            else:
                logit = torch.zeros_like(out)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder
                h_k = self.encoder_k(im)
                k = self.head_k(h_k)
                k = torch.tanh(k / np.sqrt(self.dim))
                k_1, k_2 = k[:batch_size], k[batch_size:]
                k = torch.cat([k_2, k_1], dim=0)

                if self.VI:
                    logit_k = self.prediction_k(h_k) / np.sqrt(self.dim)
                else:
                    logit_k = torch.zeros_like(out)

            logit_1, logit_2 = logit[:batch_size], logit[batch_size:]
            logit_k_1, logit_k_2 = logit_k[:batch_size], logit_k[batch_size:]
            logit = torch.cat([logit_2, logit_1], dim=0)
            logit_k = torch.cat([logit_k_2, logit_k_1], dim=0)
            keys = self.queue.clone().detach()

            # Apply symmetric noise
            m = (torch.rand_like(out) > self.p).to(torch.float32)
            m = 2 * (m - 0.5)
            out_drop = m * out

            l_pos = torch.einsum('nc,nc->n', [out_drop, k]).unsqueeze(-1)
            log_sim_matrix = torch.mm(out_drop, keys)
            log_sim_matrix = torch.cat([l_pos, log_sim_matrix], dim=1) / self.temperature
            m = log_sim_matrix.max(dim=-1)[0]
            if self.VI:
                q_k = logit_k.sigmoid()
                log_pos_sim = (torch.sum(out_drop * logit_k
                                         + torch.log(q_k * (1 - q_k)) - np.log(1/4), dim=-1)) / 2
                loss = (- log_pos_sim
                        + torch.logsumexp(log_sim_matrix, dim=-1))

                q = logit.sigmoid()
                KL = (torch.sum(out_drop.detach() * logit
                                + torch.log(q * (1 - q)) - np.log(1/4), dim=-1)) / 2
                loss = loss - KL
            else:
                log_pos_sim = l_pos
                loss = (- log_pos_sim / self.temperature
                        + torch.log(torch.exp(log_sim_matrix - m.unsqueeze(1)).sum(dim=-1)) + m)

            labels = torch.zeros(batch_size, dtype=torch.long).cuda()
            self._dequeue_and_enqueue(k)

            classifier_output = self.classifier(h_1.detach())

            return loss, out[:batch_size], log_sim_matrix, labels, keys, logit_k_2, logit_2, l_pos, k_2, classifier_output

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
