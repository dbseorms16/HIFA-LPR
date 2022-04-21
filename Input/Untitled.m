factor = 5.0;
path = "./x"+3+"_UFPR/";
savepath = "./x"+factor+"mat/";
factor=1.9
folder = dir (path);

list = size(folder)

for i = 3:list 
 
    pic = path+folder(i).name;
    
    disp(pic)
    img = imread(pic);
    %img = imresize(img, 0.0, 'bicubic');
    
    img = imresize(img, factor, 'bicubic');
    imwrite(img,"./x5mat/"+folder(i).name);
    %imwrite(img, savepath+folder(i).name);
    
end