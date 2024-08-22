from img_cap_engine.model.utils import get_lr
import torch

class Config:
    def __init__(self):
        # Data parameters
        self.data_dir = '/kaggle/working'
        self.img_dir = '/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images'
        
        # Model parameters
        self.v_size = None  # This will be set later based on the tokenizer
        self.n_emb = 512
        self.n_head = 16
        self.h_size = self.n_emb // self.n_head
        self.n_block = 10
        self.exp_fac = 6
        self.max_seq_len = 1024
        self.d_rate = 0.0
        self.p_size = 16
        self.im_size = 225
        self.c_dim = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learning parameters
        self.checkpoint_path = '/kaggle/working/ic_checkpoints.pth.tar'
        self.tok_in_batch = 1000
        self.max_b_size = 32
        self.batch_in_step = 25000 // self.tok_in_batch
        self.n_steps = 1000000
        self.max_iters = 20
        self.warmup_steps = 8000
        self.step = 1
        self.lr = get_lr(step=self.step, n_emb=self.n_emb, warmup_steps=self.warmup_steps)
        self.start_epoch = 0
        self.betas = (0.9, 0.98)
        self.eps = 1e-9
        self.weight_decay = 1e-4
        self.grad_clip = 1.0
        self.label_smoothing = 0.1
        self.cudnn_benchmark = False

    def set_vocab_size(self, tokenizer):
        self.v_size = len(tokenizer.get_vocab())

    def __repr__(self):
        return (f"<Config data_dir={self.data_dir}, img_dir={self.img_dir}, checkpoint_path={self.checkpoint_path}, "
                f"v_size={self.v_size}, n_emb={self.n_emb}, n_head={self.n_head}, h_size={self.h_size}, "
                f"n_block={self.n_block}, exp_fac={self.exp_fac}, max_seq_len={self.max_seq_len}, "
                f"d_rate={self.d_rate}, p_size={self.p_size}, im_size={self.im_size}, c_dim={self.c_dim}, "
                f"tok_in_batch={self.tok_in_batch}, max_b_size={self.max_b_size}, batch_in_step={self.batch_in_step}, "
                f"n_steps={self.n_steps}, max_iters={self.max_iters}, warmup_steps={self.warmup_steps}, "
                f"lr={self.lr:.2e}, start_epoch={self.start_epoch}, betas={self.betas}, eps={self.eps}, "
                f"weight_decay={self.weight_decay}, grad_clip={self.grad_clip}, label_smoothing={self.label_smoothing}, "
                f"device={self.device}, cudnn_benchmark={self.cudnn_benchmark}>")
