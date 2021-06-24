#!/usr/bin/env python
# coding: utf-8

# # Code

# ![example](./images/example1.png) 

# ## Motion Flow berechnen

# In[1]:


import caffe
import sys, os
import numpy as np
import scipy.io
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import timeit

model = 'model/fcn_blur2mflow.caffemodel'
img_mean = (111.64, 113.53, 93.96)
net = caffe.Net('model/fcn_blur2mflow.prototxt', model, caffe.TEST)

le_u = LabelEncoder()
le_v = LabelEncoder()
le_u.classes_ = np.loadtxt('model/labels_u.txt')
le_v.classes_ = np.loadtxt('model/labels_v.txt')

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open(sys.argv[1])
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= img_mean
in_ = in_.transpose((2,0,1))

# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
mf1 = net.blobs['score1'].data[0].argmax(0)
mf2 = net.blobs['score2'].data[0].argmax(0)

mf1 = le_u.inverse_transform(mf1.flatten()).reshape(mf1.shape)
mf2 = le_v.inverse_transform(mf2.flatten()).reshape(mf2.shape)

mfmap = np.stack((mf1, mf2)).transpose((1, 2, 0))


scipy.io.savemat(sys.argv[2], {'mfmap': mfmap})


# ## Motion Flow darstellen

# In[ ]:


get_ipython().run_line_magic('draw', 'motion field over images')
function im_mfmap = draw_mfmap(img, mag, ori)
get_ipython().run_line_magic('img', ': blurred image.')
get_ipython().run_line_magic('[mag,', 'ori]: motion flow in "magnitude + orientation" format.')
[r,c,d] = size(img);
inte = 21;
if d > 1
    img = double((rgb2gray(uint8(img)))) / 255;
end
im_mfmap(:,:,1) = 0.7 * 1 + 0.4 * img;
im_mfmap(:,:,2) = 0.7 * 1 + 0.4 * img;
im_mfmap(:,:,3) = 0.7 * 1 + 0.4 * img;
ori = 90 - ori;
rec_wid = 1;
for i = inte : inte : r - inte
    for j = inte : inte : c - inte
        l = max(mag(i, j), 1);
        o = ori(i, j);

        ft = ((fspecial('motion', l, o)));
        kkk = fspecial('average', 2);
        ft = conv2(ft, kkk, 'same');
        
        
        ft = ft / max(ft(:));
        [w,h] = size(ft);
        
        [xs, ys] = find(ft > 0);
        ids_ker = sub2ind([w,h], xs, ys);
        xs = xs - (w+1) / 2;
        ys = ys - (h+1) / 2;
        ids_img = sub2ind([r,c], i + xs, j + ys);
        
        im_mfmap(ids_img) = 1 * ft(ids_ker) + (1 - ft(ids_ker)) .* im_mfmap(ids_img);
        im_mfmap(ids_img + r * c) = (1 - ft(ids_ker)) .* im_mfmap(ids_img + r * c);
        im_mfmap(ids_img + 2 * r * c) = (1 - ft(ids_ker)) .* im_mfmap(ids_img + 2 * r * c);
        
        get_ipython().run_line_magic('draw', 'a rectangle around the centered pixel')
        if(0)
            for pp = i - rec_wid : i +  rec_wid
                for qq = j - rec_wid : j + rec_wid
                    imMotion(pp, qq, 1) = 0;
                    imMotion(pp, qq, 2) = 0;
                    imMotion(pp, qq, 3) = 1;  
                end
            end
        end
        im_mfmap(i, j, 1) = 0;
        im_mfmap(i, j, 2) = 0.5;
        im_mfmap(i, j, 3) = 0;
    end
end


# ![example1_hmap](./images/example1_hmap.png)

# ## Deconvolution

# In[ ]:


function x_est = nbd_single(y, mfmap)
get_ipython().run_line_magic('y', ': input blurred image')
get_ipython().run_line_magic('mfmap', ': input motion flow map')

mu = mfmap(:,:,1);
mv = mfmap(:,:,2);

[mag, ori]= motion2magori(-mv, mu);

params.alpha = 1/2;
params.mu = 0;
params.maxIter_out = 1;
params.maxIter_in = 5;
params.useGPU = 0;

kernelInit.mlhmag = mag;
kernelInit.mlhori = ori;

[x_est] = fast_deconv_nonUniform_gmmprior(y, y, [], params, kernelInit);

end


# ![example1_result](./images/example1_result.png)
