from .pvnet.tools.demo import *

net = Resnet18_8s(ver_dim=vote_num * 2, seg_dim=2)
net = NetWrapper(net).cuda()
net = DataParallel(net)

optimizer = optim.Adam(net.parameters(), lr=train_cfg['lr'])
model_dir = os.path.join(cfg.MODEL_DIR, "cat_demo")

class LitPVNetls(pl.LightningModule):
    def __init__(self, cfg, lr=0.02, load=True):
        super().__init__()
        net = Resnet18_8s(ver_dim=vote_num * 2, seg_dim=2)
        self.net = NetWrapper(net).cuda()
        self.fg
        self.lr = lr
        self.optim = optimizer
        self.model_dir = model_dir
        if load:
            load_model(self.net.module.net, self.optimizer, self.model_dir, -1)

    def forward(self):
        data, points_3d, bb8_3d = read_data()
        image, mask, vertex, vertex_weights, pose, corner_target = [d.unsqueeze(0).cuda() for d in data]
        return net(image, mask, vertex, vertex_weights)

    def tain(self, data, points_3d, bb8_3d):
        seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall = self.forward(image, mask, vertex, vertex_weights)

    def configure_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=self.lr)