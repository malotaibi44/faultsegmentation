from yacs.config import CfgNode as cn
import os 
_c = cn()

_c.ckpt_dir = ''
_c.device = 'cuda'
_c.seed = None
_c.print_freq = 2

_c.dataset = cn()
#the folder for the masks labels (under it should be folders for each novice/practitioner, and the expert)
_c.dataset.root = '/home/hice1/malotaibi44/scratch/segformer/Fault segmentations'
#folder of images but not used, 
_c.dataset.data_path =  '/home/hice1/malotaibi44/scratch/segformer/NeurIPS2024_SegFaults-main/images/images'
_c.dataset.datalist = ''
_c.dataset.annot_groups = os.listdir('Fault segmentations')[1:]
_c.dataset.labelset = 'certain-uncertain'
_c.dataset.pad_height = 0
_c.dataset.pad_width = 0

_c.dataloader = cn()
_c.dataloader.num_workers = 0 # of subprocesses to use for data loading

_c.trainset = cn()
_c.trainset.name = 'random'
_c.trainset.setting = 'SingleAnnot'
_c.trainset.annot_group = ['expert']
_c.trainset.singleannot = cn()
_c.trainset.singleannot.annotID = ''

_c.model = cn()
_c.model.backbone = cn()
_c.model.backbone.arch = ''
_c.model.backbone.in_planes = 1

_c.solver = cn()
_c.solver.num_epoch = 0
_c.solver.lr = 0.01
_c.solver.batch_size = 64
_c.solver.optimizer = ''
_c.solver.loss = ''
_c.solver.weight_decay = 5e-4

_c.test = cn()
_c.test.inference = False
_c.test.batch_size = 64
_c.test.metric = ''
