# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:48:14 2020

@author: user
"""
import os
import argparse
from PIL import Image, ImageOps

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return img
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

parser = argparse.ArgumentParser(description='PIL save')
parser.add_argument('--upscale_factor', type =int, default=2, help="scale factor")
parser.add_argument('--folder', type =int, default=2, help="scale factor")

parser.add_argument('--path', default="./Input/bicubic/",  help="put results here")
parser.add_argument('--HR', default="./Input/validation/", help="put HR images here")

#parser.add_argument('--W', type =int, default=178, help="pic's width")
#parser.add_argument('--H', type =int, default=218, help="pic's height")
parser.add_argument('--filename', type =str, default='name', help="pic's name")

opt = parser.parse_args()

def saving(name, loadimgpath, savepath):
    
    print(loadimgpath)
    print(savepath)
    target = load_img(loadimgpath)
    
    input1 = target.resize((int(target.size[0]/opt.upscale_factor),int(target.size[1]/opt.upscale_factor)), Image.BICUBIC)       
    
    bicubic = rescale_img(input1, opt.upscale_factor)
    input1.save(savepath+name)
    bicubic.save(savepath+'bicubic/'+name)

def search(txt):
    typeoffile = ['.jpg','.png','.bmp']
    print(txt)
    
    for str in typeoffile:
        if txt.find(str)!=-1: 
            return False
            
    return True

def calALL():
    
    path = opt.HR
 
    file_list = os.listdir(path)
 
   
    for img in file_list:
        
        if not(search(img)):
            newPath = path+img 
            saving(img, newPath, opt.path)     

def justone():

    target = load_img(opt.HR+opt.filename+'.jpg')
    print(target.size[0])
    print(target.size[1])

    input1 = target.resize((int(target.size[0]/opt.upscale_factor),int(target.size[1]/opt.upscale_factor)), Image.BICUBIC)       

    bicubic = rescale_img(input1, opt.upscale_factor)
        
    input1.save(opt.path+opt.filename+'.jpg')

calALL()