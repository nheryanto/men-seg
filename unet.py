import torch
import torch.nn as nn

class SingleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, drop_rate=None, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=(kernel_size - 1) // 2, dilation=1, bias=True)
        self.conv_bn = nn.BatchNorm2d(out_channels)
        # self.nonlin = nn.LeakyReLU(inplace=True)
        self.nonlin = nn.ReLU(inplace=True)
        if drop_rate is not None:
            self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        
    def forward(self, x):
        x = self.conv(x)
        x = self.conv_bn(x)
        x = self.nonlin(x)
        if self.drop_rate is not None:
            x = self.dropout(x)
        return x
    
class StackedConvBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, drop_rate, n_conv_per_stage=2):
        super().__init__()
        self.convs = nn.Sequential(
            SingleConvBlock(in_channels, out_channels, kernel_size, drop_rate, stride),
            *[SingleConvBlock(out_channels, out_channels, kernel_size, drop_rate)
              for i in range(1, n_conv_per_stage)]
        )
        
    def forward(self, x):
        return self.convs(x)
    
class UNetEncoder(nn.Module):
    def __init__(self, n_stages, in_channels, out_channels, kernel_sizes, strides, drop_rate=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.drop_rate = drop_rate
        
        stages = []
        for s in range(n_stages):
            stages.append(StackedConvBlocks(
                in_channels, out_channels[s], kernel_sizes[s], strides[s], drop_rate
            ))
            in_channels = out_channels[s]
        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        skips = []
        for s in self.stages:
            x = s(x)
            skips.append(x)
        return skips

class UNetDecoder(nn.Module):
    def __init__(self, encoder, num_classes, deep_supervision):
        super().__init__()
       
        stages = []
        transpose_conv = []
        seg_conv = []

        n_stages_encoder = len(encoder.out_channels)
        # upscaling
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.out_channels[-s]
            input_features_skip = encoder.out_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            transpose_conv.append(nn.ConvTranspose2d(
                input_features_below, input_features_skip,
                stride_for_transpconv, stride_for_transpconv 
            ))

            stages.append(StackedConvBlocks(
                2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.drop_rate
            ))

            if deep_supervision:
                seg_conv.append(nn.Conv2d(
                    input_features_skip, num_classes, 1, 1
                ))
            elif s == (n_stages_encoder - 1):
                seg_conv.append(nn.Conv2d(
                    input_features_skip, num_classes, 1, 1
                ))
        
        self.encoder = encoder
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        self.transpose_conv = nn.ModuleList(transpose_conv)
        self.stages = nn.ModuleList(stages)
        self.seg_conv = nn.ModuleList(seg_conv)
        
    def forward(self, skips):
        # input bottle-neck
        low_res_input = skips[-1]
        
        seg_outputs = []
        
        # do not confuse: len(self.stages) != len(n_stages_encoder)
        for s in range(len(self.stages)):
            
            # upscale
            x = self.transpose_conv[s](low_res_input)
            
            # concatenate skip connections
            x = torch.cat([x, skips[-(s + 2)]], dim=1)
            
            # conv blocks
            x = self.stages[s](x)
            
            # deep supervision
            if self.deep_supervision:
                seg_outputs.append(self.seg_conv[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_conv[-1](x))
                
            # update input
            low_res_input = x
        
        seg_outputs = seg_outputs[::-1]
        if self.deep_supervision:
            r = seg_outputs
        else:
            r = seg_outputs[0]
        return r
    
class UNet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
            
    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
    
    def set_deep_supervision_enabled(self, enabled):
        self.decoder.deep_supervision = enabled

class BBConv(nn.Module):
    def __init__(self, in_channels, out_channels, pool_ratio, stride):
        super().__init__()
        # self.mp = nn.MaxPool2d(pool_ratio)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
    def forward(self, x):
        # x = self.mp(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x
    
class BBUNetDecoder(nn.Module):
    def __init__(self, encoder, bb_pool_ratio, bb_stride, num_classes, deep_supervision):
        super().__init__()
       
        stages = []
        bb_conv = []
        transpose_conv = []
        seg_conv = []

        n_stages_encoder = len(encoder.out_channels)
        # upscaling (bottom-up)
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.out_channels[-s]
            input_features_skip = encoder.out_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            
            bb_conv.append(BBConv(
                1, input_features_skip, bb_pool_ratio[-s], bb_stride[-s]
            ))

            transpose_conv.append(nn.ConvTranspose2d(
                input_features_below, input_features_skip,
                stride_for_transpconv, stride_for_transpconv 
            ))

            stages.append(StackedConvBlocks(
                2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, encoder.drop_rate
            ))
            
            if deep_supervision:
                seg_conv.append(nn.Conv2d(
                    input_features_skip, num_classes, 1, 1
                ))
            elif s == (n_stages_encoder - 1):
                seg_conv.append(nn.Conv2d(
                    input_features_skip, num_classes, 1, 1
                ))
        
        self.encoder = encoder
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        self.bb_conv = nn.ModuleList(bb_conv)
        self.transpose_conv = nn.ModuleList(transpose_conv)
        self.stages = nn.ModuleList(stages)
        self.seg_conv = nn.ModuleList(seg_conv)
        
    def forward(self, skips, bb):
        # input bottle-neck
        low_res_input = skips[-1]
        
        seg_outputs = []
        
        # do not confuse: len(self.stages) != len(n_stages_encoder)
        for s in range(len(self.stages)):
            
            # bounding box encoder
            bb_encoder = self.bb_conv[s](bb)
            skips_bb = skips[-(s + 2)] * bb_encoder
            
            # upscale
            x = self.transpose_conv[s](low_res_input)
            
            # concatenate skip connections
            x = torch.cat([x, skips_bb], dim=1)
            
            # conv blocks
            x = self.stages[s](x)
            
            # deep supervision
            if self.deep_supervision:
                seg_outputs.append(self.seg_conv[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_conv[-1](x))
                
            # update input
            low_res_input = x
        
        seg_outputs = seg_outputs[::-1]
        if self.deep_supervision:
            r = seg_outputs
        else:
            r = seg_outputs[0]
        return r

class BBUNet(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
            
    def forward(self, x, bb):
        skips = self.encoder(x)
        return self.decoder(skips, bb)
    
    def set_deep_supervision_enabled(self, enabled):
        self.decoder.deep_supervision = enabled