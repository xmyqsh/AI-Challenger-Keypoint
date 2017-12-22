--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  AI dataset loader (from  (Newell))
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/posetransforms'


-------------------------------------------------------------------------------
-- Helper Functions
-------------------------------------------------------------------------------
local getTransform = t.getTransform
local transform = t.transform
-- local crop = t.crop2
local crop = t.crop
local drawGaussian = t.drawGaussian
local shuffleLR = t.shuffleLR
local flip = t.flip
local colorNormalize = t.colorNormalize

-------------------------------------------------------------------------------
-- Create dataset Class
-------------------------------------------------------------------------------

local M = {}
local AIDataset = torch.class('resnet.AIDataset', M)

function AIDataset:__init(imageInfo, opt, split)
  assert(imageInfo[split], split)
  self.imageInfo = imageInfo[split]
  self.split = split
  -- Some arguments
  self.inputRes = opt.inputRes
  self.outputRes = opt.outputRes
  -- Options for augmentation
  self.scaleFactor = opt.scaleFactor
  self.rotFactor = opt.rotFactor
  self.dataset = opt.dataset
  self.nStack = opt.nStack
  self.meanstd = torch.load('gen/AI/meanstd.t7')
  self.nGPU = opt.nGPU
  self.batchSize = opt.batchSize
  self.minusMean = opt.minusMean
  self.gsize = opt.gsize
  self.bg = opt.bg
  self.rotProbab = opt.rotProbab
  self.dataAug = opt.dataAug
  self.focus = opt.focus

  -- Key Point standard Variance
  --[[
  self.delta = {0.02776304,  0.03030456,  0.0211533 ,  0.02835418,  0.02995782,
           0.02804288,  0.07819284,  0.07373882,  0.03963606,  0.07687942,
           0.06824636,  0.04830162,  0.02582912,  0.02472346}
  --]]
  -- delta2 = np.sqrt(2)*delta
  self.delta_2 = {0.03926287,  0.04285712,  0.02991528,  0.04009887,  0.04236676,
                  0.03965862,  0.11058137,  0.10428244,  0.05605385,  0.10872392,
                  0.09651493,  0.06830881,  0.03652789,  0.03496425}
  -- alpha = np.sqrt(-np.log(0.5:0.05:0.95))
  self.alpha = {0.83255461,  0.7731992 ,  0.71472066,  0.65634055,  0.59722269,
                0.53636002,  0.47238073,  0.40313637,  0.32459285}
end

function AIDataset:get(i, scaleFactor)
   local scaleFactor = scaleFactor or 1
   local img = image.load(paths.concat('data/AI/images', self.imageInfo.data['images'][i]))

   -- Generate samples
   local pts = self.imageInfo.labels['part'][i]
   local c = self.imageInfo.labels['center'][i]
   local s = self.imageInfo.labels['scale'][i]*scaleFactor

   -- For single-person pose estimation with a centered/scaled figure
   local nParts = pts:size(1)
   --[[
   if not pcall(function() crop(img, c, s, 0, self.inputRes) end) then
     print("i")
     print(i)
     print("c")
     print(c)
     print("s")
     print(s)
     print("image_path")
     print(self.imageInfo.data['images'][i])
   end
   --]]
   local inp = crop(img, c, s, 0, self.inputRes)
   -- if not pcall(function() local inp = crop(img, c, s, 0, self.inputRes) end) then
     -- print("i")
     -- print(i)
   -- end
 if self.focus then
   outTable = {}
   local nStack = self.nStack
   for n = 1,nStack do
     local out = self.bg == 'true' and torch.zeros(nParts+1, self.outputRes, self.outputRes) 
                                or torch.zeros(nParts, self.outputRes, self.outputRes)

     for i = 1,nParts do
       if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
           sqrtArea = 200 * math.sqrt(s[1]*s[2])
           d = self.delta_2[i] * self.alpha[n] * sqrtArea
           h = 200 * s
           -- print("h")
           -- print(h)
           D = h
           D[1] = math.floor(d * self.outputRes / h[1])
           D[2] = math.floor(d * self.outputRes / h[2])
           -- print("D")
           -- print(D)
           drawGaussian(out[i], transform(torch.add(pts[i],1), c, s, 0, self.outputRes), D)
       end
     end

     if self.bg == 'true' then
        out[nParts+1], _ = torch.max(out:sub(1, nParts, 1, self.outputRes, 1, self.outputRes), 1)
     end

     table.insert(outTable, out)

   end
   out = outTable
 else
   n = 1
   out = self.bg == 'true' and torch.zeros(nParts+1, self.outputRes, self.outputRes) 
                              or torch.zeros(nParts, self.outputRes, self.outputRes)
   for i = 1,nParts do
     if pts[i][1] > 0 then -- Checks that there is a ground truth annotation
         sqrtArea = 200 * math.sqrt(s[1]*s[2])
         d = self.delta_2[i] * self.alpha[n] * sqrtArea
         h = 200 * s
         -- print("h")
         -- print(h)
         D = h
         D[1] = math.ceil(d * self.outputRes / h[1])
         D[2] = math.ceil(d * self.outputRes / h[2])
         -- print("D")
         -- print(D)
         drawGaussian(out[i], transform(torch.add(pts[i],1), c, s, 0, self.outputRes), D)
     end
   end

   if self.bg == 'true' then
      out[nParts+1], _ = torch.max(out:sub(1, nParts, 1, self.outputRes, 1, self.outputRes), 1)
   end

 end

   -- Data augmentation
   if self.dataAug then
      inp, out = self.augmentation(self, inp, out)
   end
   collectgarbage()
   return {
      input = inp,
      target = out,
      center = c,
      scale = s,
      width = img:size(3),
      height = img:size(2),
      imgPath = paths.concat('data/AI/images', self.imageInfo.data['images'][i])
   }
end

function AIDataset:size()
  if self.split == 'test' then
    return self.imageInfo.labels.nsamples
  end

  if string.find(self.split, 'det') then
    return self.imageInfo.labels.nsamples
  end
  
   local nSamples = self.imageInfo.labels.nsamples - (self.imageInfo.labels.nsamples%self.nGPU)
   nSamples = nSamples - nSamples%self.batchSize

   return nSamples
end

function AIDataset:preprocess()
   return function(img)
      if img:max() > 2 then
         img:div(255)
      end
      return self.minusMean == 'true' and colorNormalize(img, self.meanstd) or img
   end
end

function AIDataset:augmentation(input, label)
  -- Augment data (during training only)
  if self.split == 'train' then
      local s = torch.randn(1):mul(self.scaleFactor):add(1):clamp(1-self.scaleFactor,1+self.scaleFactor)[1]
      local r = torch.randn(1):mul(self.rotFactor):clamp(-2*self.rotFactor,2*self.rotFactor)[1]

      -- Color
      input[{1, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      input[{2, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)
      input[{3, {}, {}}]:mul(torch.uniform(0.8, 1.2)):clamp(0, 1)

      -- Scale/rotation
      if torch.uniform() <= (1-self.rotProbab) then r = 0 end
      local inp,out = self.inputRes, self.outputRes
      input = crop(input, torch.Tensor({(inp+1)/2,(inp+1)/2}), torch.Tensor({inp*s/200,inp*s/200}), r, inp)
      if torch.type(label) == 'table' then
          for n = 1, #label do
              label[n] = crop(label[n], torch.Tensor({(out+1)/2,(out+1)/2}), torch.Tensor({out*s/200,out*s/200}), r, out)
          end
      else
          label = crop(label, torch.Tensor({(out+1)/2,(out+1)/2}), torch.Tensor({out*s/200,out*s/200}), r, out)
      end

      -- Flip
      if torch.uniform() <= .5 then
          input = flip(input)
          if torch.type(label) == 'table' then
              for n = 1, #label do
                  label[n] = flip(shuffleLR(label[n], self.dataset))
              end
          else
              label = flip(shuffleLR(label, self.dataset))
          end
      end
  end

  return input, label
end

return M.AIDataset
