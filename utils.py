import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch import mm

def norm(img, vgg=False):
    if vgg:
        # normalize for pre-trained vgg model
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        # normalize [-1, 1]
        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    return transform(img)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def denorm(img, vgg=False):
    if vgg:
        transform = transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                                         std=[4.367, 4.464, 4.444])
        return transform(img)
    else:
        out = (img + 1) / 2
        return out.clamp(0, 1)
        
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
def count_parameters(self, model):
        # DRN's Scale factor
        # ratio = ratio
        # File route of Modified parameters.txt, Scale에 따라서 각각 저장된다.
        route = './new/Modified parameters.txt'
        param_file = open(route,'w')
        
        # Part 별로 total parameters summation
        sub_mean = 0
        head = 0
        down_block = 0
        up_blocks_0 = 0
        up_blocks_1 = 0
        tail = 0
        add_mean = 0
        
        for name, p in model.named_parameters():
            weight_name = str(name)
            param = p.numel()
            if 'up_blocks.0' in weight_name:
                up_blocks_0 += param  
            elif 'up_blocks.1' in weight_name:
                up_blocks_1 += param
            elif 'tail' in weight_name:
                tail += param
            elif 'head' in weight_name:
                head += param
            elif 'down' in weight_name:
                down_block += param
            elif 'sub_mean' in weight_name:
                sub_mean += param
            else:
                add_mean += param
                
            param_file.write(f'name:{name} \n')
            param_file.write(f'param.shape:{p.shape} \n')
            param_file.write(f'param.shape:{p.numel()} \n')
            param_file.write('======================================\n')
        
        param_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_file.write(f'\nThe number of parameters : {param_sum}')
        param_file.write(f' - about {param_sum / 1000 ** 2:.2f}M\n')
        param_file.write(f'\nParameters of sub_mean : {sub_mean}\n')
        param_file.write(f'\nParameters of head : {head}\n')
        param_file.write(f'\nParameters of down_block : {down_block}\n')
        param_file.write(f'\nParameters of up_block_0 : {up_blocks_0}')
        param_file.write(f' - about {up_blocks_0 / 1000 ** 2:.2f}M\n')
        param_file.write(f'\nParameters of up_block_1 : {up_blocks_1}')
        param_file.write(f' - about {up_blocks_1 / 1000 ** 2:.2f}M\n')
        param_file.write(f'\nParameters of tail : {tail}\n')
        param_file.write(f'\nParameters of add_mean : {add_mean}\n')
        param_file.write(f'\nFrom sub_mean to add_mean\n')
        param_file.write(f'\n{sub_mean} {head} {down_block} {up_blocks_0} {up_blocks_1} {tail} {add_mean}\n')
        param_file.close()
        return param_sum