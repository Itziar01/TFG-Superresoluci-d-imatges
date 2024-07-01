import torch
import torch.nn as nn
    
class SRCNN(nn.Module):
    '''
    CNN network
    @num_feat (int) number of channels of original image.
    @large_kernel, small_kernel (int) size for diferents kernels.
    @upscale_factor (int) upsacel factor for output image.
    '''
    def __init__(self, num_feat : int, large_kernel : int, small_kernel : int, upscale_factor : int) -> None:
        super(SRCNN, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(num_feat, 64, large_kernel, padding=large_kernel // 2), 
            nn.ReLU()
        )
        
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, small_kernel, padding=small_kernel // 2),
            nn.ReLU()
        ) 
    

        self.reconstruction = nn.Sequential(
            nn.Conv2d(32, num_feat, small_kernel, padding=small_kernel // 2),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = self.map(x)
        sr_image = self.reconstruction(x)
        
        return sr_image
