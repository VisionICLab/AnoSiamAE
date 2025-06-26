from torch import nn
import src.loss.reconstruction as rec
import src.loss.perceptual as perc
import src.loss.adversarial as adv
    
class GeneratorLossManager(nn.Module):
    def __init__(self, params, device):
        super(GeneratorLossManager, self).__init__()
        self.params = params
        self.device = device
        self.losses = []
        self.weights = params.WEIGHTS
        self.reduction = self.params.REDUCTION
        for loss_name in self.params.NAME:
            self.losses.append(self.init_loss(loss_name))
        
    def forward(self, img, rec):
        dict={}
        for i,name in enumerate(self.params.NAME):
            dict[name] = self.weights[i]*self.losses[i](img, rec)
        return dict
    
    def init_loss(self, name):
        if name == "MAE" or name == "L1":
            return rec.MAELoss(self.device, reduction=self.params.REDUCTION)
        elif name == "MSE" or name == "L2":
            return rec.MSELoss(self.device, reduction=self.params.REDUCTION)
        elif name == "CrossEntropy":
            return rec.CrossEntropyLoss(self.device, reduction=self.params.REDUCTION)
        elif name == "SSIM":
            return rec.SSIMLoss(self.device, reduction=self.params.REDUCTION)
        elif name == "MS_SSIM":
            return rec.MS_SSIMLoss(self.device, reduction=self.params.REDUCTION)
        elif name == "LPIPS":
            return perc.LPIPSLoss(self.device, net_type=self.params.NET_TYPE, reduction=self.params.REDUCTION)
        elif name == "PerceptualVGG":
            return perc.PerceptualVGGLoss(self.device, reduction=self.params.REDUCTION, targets=self.params.LAYERS)
        elif name == "RelativePerceptualVGG":
            return perc.RelativePerceptualVGGLoss(self.device, reduction=self.params.REDUCTION, targets=self.params.LAYERS)

    
class AdversarialLossManager(nn.Module):
    def __init__(self, params, device):
        super(AdversarialLossManager, self).__init__()
        self.params = params
        self.device = device
        self.losses = []
        self.weights = params.WEIGHTS
        for loss_name in self.params.NAME:
            self.losses.append(self.init_loss(loss_name))
        
    def forward(self, outD, is_real, for_disc):
        l = 0
        for i,loss in enumerate(self.losses):
            l+=self.weights[i]*loss(outD, is_real, for_disc)
        return l
    
    def init_loss(self, name):
        if name == "PatchAdversarial":
            return adv.PatchAdversarialLoss(self.device, reduction=self.params.REDUCTION)
        if name == "BinaryAdversarial":
            return adv.BinaryAdversarialLoss(self.device, reduction=self.params.REDUCTION)
