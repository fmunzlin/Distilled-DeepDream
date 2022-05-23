import time
import torchvision.models as TM
import torchvision.transforms.functional
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.nn import init
import train_network
import os
from os import path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import utils.utils as utils
from utils.constants import *
import torch
from models.definitions import vggs
import tqdm
import argparse
import torchvision.utils as vutils
parser = argparse.ArgumentParser()

#general settings
parser.add_argument('--d_id', type=int, default=0, help='cuda device id')
parser.add_argument('--iterations', type=int, default=100000, help='cuda device id')
parser.add_argument('--start_se', type=int, default=50000, help='cuda device id')
parser.add_argument('--start_sd', type=int, default=75000, help='cuda device id')
parser.add_argument('--decoder_steps', type=int, default=10, help='cuda device id')
parser.add_argument('--batch_size', type=int, default=2, help='cuda device id')
parser.add_argument('--save_iter', type=int, default=100, help='cuda device id')
parser.add_argument('--save_img', type=int, default=10, help='cuda device id')
parser.add_argument('--exp', type=str, default="test", help='cuda device id')
parser.add_argument('--load', type=bool, default=True, help='cuda device id')
parser.add_argument('--use_SE', type=bool, default=True, help='cuda device id')

class Data(Dataset):
    def __init__(self, path=osp.join("data", "mscoco", "images")):
        self.path = path
        self.images = os.listdir(self.path)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.augment = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        self.denorm = transforms.Normalize(-torch.as_tensor(self.mean) / torch.as_tensor(torch.as_tensor(self.std)),
                                           1 / (torch.as_tensor(self.std) + 1e-7))

        self.mobile = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

    # we resize an image to have to smallest side of an image to fit a fix number of pixels
    def resize_to_min(self, resize_to, image):
        width, height = image.size
        if height <= width:
            factor = resize_to / height
        else:
            factor = resize_to / width
        return self.resize_to_factor(image, factor)

    # resize an image to a given factor
    # afterwards we multiply the factor to both width and height
    def resize_to_factor(self, img, factor):
        if isinstance(factor, list):
            return img.resize((int(img.width * factor[0]), int(img.height * factor[0])))
        else:
            return img.resize((int(img.width * factor), int(img.height * factor)))


    def __getitem__(self, item):
        image = Image.open(osp.join(self.path, self.images[item]))
        image = self.resize_to_min(769, image)
        image = self.augment(image)
        image = transforms.CenterCrop(768)(image)
        image = image.unsqueeze(0)
        image = image.cuda()
        return image

    def __len__(self):
        return len(self.images)

    def get_rnd(self):
        return self.__getitem__(np.random.randint(0, self.__len__()))

    def get_rnd_batch(self, size):
        out = torch.Tensor().cuda()
        for i in range(size):
            out = torch.cat([out, self.get_rnd()], dim=0)
        return out


class Auto_encoder(nn.Module):
    def __init__(self, args):
        super(Auto_encoder, self).__init__()
        self.args = args
        # self.encoder = nn.Sequential(*list(TM.vgg16(pretrained=True).features.children())[:-7])
        self.encoder = vggs.Vgg16Experimental(requires_grad=False, show_progress=True)
        self.decoder = Decoder()

        self.small_encoder = SmallEncoder5_16x_aux()
        self.small_decoder = SmallDecoder5_16x()
        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters'): layer.reset_parameters()
        self.mode = ""
        self.switch_mode()


    def switch_mode(self):
        if self.mode == "":
            self.set_training(self.decoder, [self.encoder, self.small_encoder, self.small_decoder])
            self.mode = "base"
        elif self.mode == "base":
            self.set_training(self.small_encoder, [self.encoder, self.decoder, self.small_decoder])
            self.mode = "se"
        elif self.mode == "se":
            self.set_training(self.small_decoder, [self.encoder, self.small_encoder, self.decoder])
            self.mode = "sd"
        else:
            raise("Mode not implemented")

    def set_training(self, train_model, eval_models):
        train_model.train()
        for param in train_model.parameters(): param.requires_grad = True

        for model in eval_models:
            model.eval()
            for param in model.parameters(): param.requires_grad = False

    def forward(self, input_tensor, old_acid):
        if self.mode == "base":
            input_feat = self.encoder.forward(input_tensor)
            styled = self.decoder.forward(input_feat[0])
        elif self.mode == "se":
            input_feat = self.small_encoder.forward_aux(input_tensor)
            styled = self.decoder.forward(input_feat[-1])
        elif self.mode == "sd":
            input_feat = self.small_encoder.forward(input_tensor)
            styled = self.small_decoder.forward(input_feat)
        else:
            raise("Mode not implemented")

        styled_feat, _, styled_relus = self.encoder.forward(styled)
        acid_feat, _, acid_relus = self.encoder.forward(old_acid)

        # pixel loss
        loss = nn.MSELoss()(styled, old_acid)

        # perceptual loss
        for i in range(len(styled_feat)):
            loss += nn.MSELoss()(styled_feat[i], acid_feat[i])

        # knowledge transfer loss
        if self.mode == "sm":
            for i in range(len(acid_relus)):
                loss += nn.MSELoss()(acid_relus[i], styled_relus[i])

        return styled, loss

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.channel_mult = 64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, self.channel_mult*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.channel_mult*1, 3, 4, 2, 1, bias=False),
            nn.Sigmoid())

    def forward(self, feat):
        return self.deconv(feat)


class SmallEncoder5_16x_aux(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallEncoder5_16x_aux, self).__init__()
        self.fixed = fixed

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv0.requires_grad = False
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 4, 2, 0, dilation=1)
        self.conv21 = nn.Conv2d(16, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(32, 64, 3, 1, 0, dilation=1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv41 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv42 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv43 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv44 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv51 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv11_aux = nn.Conv2d(16, 64, 1, 1, 0)
        self.conv21_aux = nn.Conv2d(32, 128, 1, 1, 0)
        self.conv31_aux = nn.Conv2d(64, 256, 1, 1, 0)
        self.conv41_aux = nn.Conv2d(128, 512, 1, 1, 0)
        self.conv51_aux = nn.Conv2d(128, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if "model" in weights:
                self.load_state_dict(weights["model"])
            else:
                self.load_state_dict(weights)
            print("load model '%s' successfully" % model)

        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.conv0(y)
        y = self.relu(self.conv11(self.pad(y)))
        y = self.relu(self.conv12(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv21(self.pad(y)))
        y = self.relu(self.conv22(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv31(self.pad(y)))
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.relu(self.conv34(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv41(self.pad(y)))
        y = self.relu(self.conv42(self.pad(y)))
        y = self.relu(self.conv43(self.pad(y)))
        y = self.relu(self.conv44(self.pad(y)))
        y = self.pool(y)
        y = self.relu(self.conv51(self.pad(y)))
        return y

    def forward_aux(self, input):
        out0 = self.conv0(input)
        out11 = self.relu(self.conv11(self.pad(out0)))
        out12 = self.relu(self.conv12(self.pad(out11)))
        out12 = self.pool(out12)
        out21 = self.relu(self.conv21(self.pad(out12)))
        out22 = self.relu(self.conv22(self.pad(out21)))
        out22 = self.pool(out22)
        out31 = self.relu(self.conv31(self.pad(out22)))
        out32 = self.relu(self.conv32(self.pad(out31)))
        out33 = self.relu(self.conv33(self.pad(out32)))
        out34 = self.relu(self.conv34(self.pad(out33)))
        out34 = self.pool(out34)
        out41 = self.relu(self.conv41(self.pad(out34)))
        out42 = self.relu(self.conv42(self.pad(out41)))
        out43 = self.relu(self.conv43(self.pad(out42)))
        out44 = self.relu(self.conv44(self.pad(out43)))
        out44 = self.pool(out44)
        out51 = self.relu(self.conv51(self.pad(out44)))
        '''
        out11_aux = self.relu(self.conv11_aux(out11))
        out21_aux = self.relu(self.conv21_aux(out21))
        out31_aux = self.relu(self.conv31_aux(out31))
        out41_aux = self.relu(self.conv41_aux(out41))
        out51_aux = self.relu(self.conv51_aux(out51))
        '''
        out11_aux = self.conv11_aux(out11)
        out21_aux = self.conv21_aux(out21)
        out31_aux = self.conv31_aux(out31)
        out41_aux = self.conv41_aux(out41)
        out51_aux = self.conv51_aux(out51)

        return out11_aux, out21_aux, out31_aux, out41_aux, out51_aux


class SmallDecoder5_16x(nn.Module):
    def __init__(self, model=None, fixed=False):
        super(SmallDecoder5_16x, self).__init__()
        self.fixed = fixed

        self.conv51 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv44 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv43 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv42 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv41 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv34 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv33 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 0, dilation=1)
        self.conv31 = nn.Conv2d(64, 32, 3, 1, 0, dilation=1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 0, dilation=1)
        self.conv21 = nn.Conv2d(32, 16, 3, 1, 0, dilation=1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 0, dilation=1)
        self.conv11 = nn.Conv2d(16, 3, 3, 1, 0, dilation=1)

        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1))

        if model:
            weights = torch.load(model, map_location=lambda storage, location: storage)
            if "model" in weights:
                self.load_state_dict(weights["model"])
            else:
                self.load_state_dict(weights)
            print("load model '%s' successfully" % model)

        if fixed:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, y):
        y = self.relu(self.conv51(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv44(self.pad(y)))
        y = self.relu(self.conv43(self.pad(y)))
        y = self.relu(self.conv42(self.pad(y)))
        y = self.relu(self.conv41(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv34(self.pad(y)))
        y = self.relu(self.conv33(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv32(self.pad(y)))
        y = self.relu(self.conv31(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv22(self.pad(y)))
        y = self.relu(self.conv21(self.pad(y)))
        y = self.unpool(y)
        y = self.relu(self.conv12(self.pad(y)))
        y = self.relu(self.conv11(self.pad(y)))
        return y

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.path = osp.join("exp", args.exp)
        os.makedirs(osp.join(self.path, "snapshots"), exist_ok=True)
        # self.dataset = Data(path=osp.join("data", "real_and_fake_face", "training_real"))
        self.dataset = Data(path=osp.join("data", "mscoco", "images"))
        self.model = Auto_encoder(self.args)
        if self.args.d_id != -1: self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(params=self.model.decoder.parameters(), lr=0.0001, betas=(0, 0.999), eps=1e-8)
        if self.args.load: self.load_checkpoint()

    def snapshot(self, image):
        image = self.dataset.denorm(image)
        grid = vutils.make_grid(image, nrow=image.size()[0])
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        image = Image.fromarray(ndarr)
        image.save(osp.join("exp", self.args.exp, "snapshots", str(self.iter_counter) + ".jpg"))

    def save_checkpoint(self):
        torch.save({"decoder": self.model.decoder,
                    "small_encoder": self.model.small_encoder,
                    "small_decoder": self.model.small_decoder}, osp.join(self.path, "checkpoint.pth"))

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(osp.join("exp", self.args.exp, "checkpoint.pth"),
                                    map_location=lambda storage, loc: storage)
            self.model.small_encoder.load_state_dict(checkpoint["small_encoder"])
            self.model.small_decoder.load_state_dict(checkpoint["small_decoder"])
            self.model.decoder.load_state_dict(checkpoint["decoder"])
        except:
            print("No checkpoint file")


    def process_acid(self, input_tensor, iteration):
        _, layer_activation, _ = self.model.encoder(input_tensor)
        # layer_activation = out[self.model.encoder.layer_names.index("relu4_3")]
        loss = torch.nn.MSELoss(reduction='mean')(layer_activation, torch.zeros_like(layer_activation))
        loss.backward()
        # Step 3: Process image gradients (smoothing + normalization)
        grad = input_tensor.grad.data

        # Applies 3 Gaussian kernels and thus "blurs" or smoothens the gradients and gives visually more pleasing results
        # sigma is calculated using an arbitrary heuristic feel free to experiment
        sigma = ((iteration + 1) / 25) * 2.0 + 0.5
        smooth_grad = utils.CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well

        # Normalize the gradients (make them have mean = 0 and std = 1)
        # I didn't notice any big difference normalizing the mean as well - feel free to experiment
        g_std = torch.std(smooth_grad)
        g_mean = torch.mean(smooth_grad)
        smooth_grad = smooth_grad - g_mean
        smooth_grad = smooth_grad / g_std

        # Step 4: Update image using the calculated gradients (gradient ascent step)
        input_tensor.data += 0.09 * smooth_grad

        # Step 5: Clear gradients and clamp the data (otherwise values would explode to +- "infinity")
        input_tensor.grad.data.zero_()
        input_tensor.data = torch.max(torch.min(input_tensor, UPPER_IMAGE_BOUND), LOWER_IMAGE_BOUND)

    def compute_acid(self, input_tensor):
        for iteration in range(25):
            h_shift, w_shift = np.random.randint(-32, 32 + 1, 2)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift)
            self.process_acid(input_tensor, iteration)
            input_tensor = utils.random_circular_spatial_shift(input_tensor, h_shift, w_shift, should_undo=True)
        return input_tensor

    def prepare_images(self, input_tensor, acid):
        normalize = transforms.Normalize(self.dataset.mean, self.dataset.std)
        i, j, h, w = transforms.RandomCrop.get_params(acid, output_size=(256, 256))

        acid = normalize(acid)
        acid = torchvision.transforms.functional.crop(acid, i, j, h, w)

        input_tensor = normalize(input_tensor)
        input_tensor = torchvision.transforms.functional.crop(input_tensor, i, j, h, w)
        return input_tensor, acid

    def train_model(self, input_tensor, old_acid):
        for i in range(self.args.decoder_steps):
            self.optimizer.zero_grad()
            train_input, train_acid = self.prepare_images(input_tensor, old_acid)
            styled, loss = self.model.forward(train_input, train_acid)
            loss.backward()
            self.optimizer.step()
            if self.iter_counter % self.args.save_img == 0:
                self.snapshot(styled)
            if self.iter_counter % self.args.save_iter == 0:
                self.save_checkpoint()


    def train(self):
        self.model.encoder.eval().cuda()
        for param in self.model.encoder.parameters(): param.require_grad = False
        self.model.decoder.train().cuda()
        for param in self.model.decoder.parameters(): param.require_grad = True

        input_tensor = self.dataset.get_rnd_batch(self.args.batch_size)
        old_acid = self.compute_acid(input_tensor).detach()
        old_input = input_tensor.detach()
        for self.iter_counter in tqdm.tqdm(range(self.args.iterations)):
            start_time = time.time()
            self.train_model(old_input, old_acid)
            print(time.time() - start_time)
            input_tensor = self.dataset.get_rnd_batch(self.args.batch_size)
            start_time = time.time()
            old_acid = self.compute_acid(input_tensor).detach()
            old_input = input_tensor.detach()
            print(time.time() - start_time)
            torch.cuda.synchronize()
            if self.iter_counter == self.args.start_se or self.iter_counter == self.args.start_sd:
                self.model.switch_mode()

def main(args):
    Trainer(args).train()

if __name__ == '__main__':
    args = parser.parse_args()
    torch.cuda.set_device(args.d_id)
    main(args)