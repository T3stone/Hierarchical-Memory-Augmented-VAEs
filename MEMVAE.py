import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class HierarchicalMemoryLayer(nn.Module):
    def __init__(self, input_dim, n_levels, num_slots_list, slot_dim_list, act=nn.LeakyReLU(), softmax_temperature=1.0):
        super(HierarchicalMemoryLayer, self).__init__()
        assert len(num_slots_list) == n_levels and len(slot_dim_list) == n_levels
        
        self.n_levels = n_levels
        self.slot_dims = slot_dim_list
        
        self.memory_banks = nn.ParameterList([
            nn.Parameter(torch.Tensor(num_slots, slot_dim)) 
            for num_slots, slot_dim in zip(num_slots_list, slot_dim_list)
            ])
        self.controllers = nn.ParameterList([
            nn.Parameter(torch.Tensor(slot_dim, num_slots)) 
            for slot_dim, num_slots in zip(slot_dim_list, num_slots_list)
            ])
        
        self.res_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(slot_dim, slot_dim),
                act,
                nn.Linear(slot_dim, slot_dim)
            ) for slot_dim in slot_dim_list
        ])

         # 初始化参数
        for param in self.memory_banks:
            torch.nn.init.xavier_normal_(param)
        for param in self.controllers:
            torch.nn.init.xavier_normal_(param)
        for mlp in self.res_mlps:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_normal_(layer.weight)
        self.projections = nn.ModuleList([nn.Identity()]) 
        for i in range(n_levels-1):
            self.projections.append(
                nn.Linear(slot_dim_list[i], slot_dim_list[i+1]) 
                if slot_dim_list[i] != slot_dim_list[i+1] 
                else nn.Identity()
            )
        
        # 输入输出投影
        self.input_proj = nn.Linear(input_dim, slot_dim_list[0])
        self.output_proj = nn.Linear(slot_dim_list[-1], input_dim)
        
        
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(num_slots)) 
            for num_slots in num_slots_list
        ])
        
        self.act = act
        self.temp = softmax_temperature

    def forward(self, x):
        # 初始投影
        refined = self.input_proj(x)  # [B, dim0]
        
        for level in range(self.n_levels):
            # 当前层参数
            memory = self.memory_banks[level]  # [num_slots, dim]
            ctrl = self.controllers[level]     # [dim, num_slots]
            bias = self.biases[level]          # [num_slots]
            
            # 注意力计算
            attn = torch.matmul(refined, ctrl) + bias.unsqueeze(0)  # [B, num_slots]
            attn = F.softmax(attn / self.temp, dim=1)
            
            # 记忆聚合
            current_mem = torch.matmul(attn, memory)  # [B, dim]
            current_mem = self.res_mlps[level](current_mem)  # 通过残差路径的MLP
            residual = refined  # 保存原始输入
            # 残差连接（保持梯度流）
            refined = refined + current_mem  
            
            # 激活函数
            refined = self.act(refined)
            
            # 投影到下一层维度
            if level < self.n_levels - 1:
                refined = self.projections[level+1](refined)  # 转换到下一层dim
        
        return self.output_proj(refined)

class LadderCompositionLayer(nn.Module):
    def __init__(self, num_inputs, nonlinearity=nn.Sigmoid(), nonlinearity_final=nn.ReLU()):
        super(LadderCompositionLayer, self).__init__()
        self.num_inputs = num_inputs
        self.nonlinearity = nonlinearity
        self.nonlinearity_final = nonlinearity_final

        # Parameters a1, a2, a3, a4
        self.a1 = nn.Parameter(torch.zeros(num_inputs))
        self.a2 = nn.Parameter(torch.ones(num_inputs))
        self.a3 = nn.Parameter(torch.zeros(num_inputs))
        self.a4 = nn.Parameter(torch.zeros(num_inputs))

        # Parameters c1, c2, c3, c4
        self.c1 = nn.Parameter(torch.zeros(num_inputs))
        self.c2 = nn.Parameter(torch.ones(num_inputs))
        self.c3 = nn.Parameter(torch.zeros(num_inputs))
        self.c4 = nn.Parameter(torch.zeros(num_inputs))

        # Parameter b1, not regularized
        self.b1 = nn.Parameter(torch.zeros(num_inputs), requires_grad=False)

    def forward(self, u, z_lat):
        sigval = self.c1 + self.c2 * z_lat
        sigval += self.c3 * u + self.c4 * z_lat * u
        sigval = self.nonlinearity(sigval)
        z_est = self.a1 + self.a2 * z_lat + self.b1 * sigval
        z_est += self.a3 * u + self.a4 * z_lat * u
        return self.nonlinearity_final(z_est)   

class Encoder(nn.Module):
    def __init__(self, num_features,hidden_dim,latent_dim):
        super(Encoder,self).__init__()
        self.linears=nn.Sequential(
            nn.Linear(num_features,hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU())
        self.latent=nn.Linear(hidden_dim,latent_dim)
        self.get_mu=nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, latent_dim))
        self.get_logvar=nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                  nn.BatchNorm1d(hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, latent_dim))
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, image):
        output=[]
        x=self.linears[0](image)
        x=self.linears[1](x)
        x=self.linears[2](x)
        output.append(x)
        temp=x
        #print(x.shape)
        x=self.linears[3](x)
        x=self.linears[4](x)
        x=self.linears[5](x)
        output.append(x)
        x=self.latent(x)
        mu=self.get_mu(temp)
        logvar=self.get_logvar(temp)
        z=self._reparameterize(mu, logvar)
        return z,mu, logvar,output
 
class Decoder(nn.Module):
    def __init__(self, n_levels, num_slots_list, slot_dim_list,latent_dim,hidden_dim,image_channels,image_size):
        super(Decoder,self).__init__()
        self.memorylayer1=HierarchicalMemoryLayer(hidden_dim, n_levels, num_slots_list, slot_dim_list, act=nn.Sigmoid(),softmax_temperature=1.0)
        self.memorylayer2=HierarchicalMemoryLayer(hidden_dim*2, n_levels, num_slots_list, slot_dim_list, act=nn.Sigmoid(), softmax_temperature=1.0)
        self.linear1=nn.Sequential(
            nn.Linear(latent_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.linear2=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
        )
        self.ladder1=LadderCompositionLayer(num_inputs=hidden_dim)
        self.ladder2=LadderCompositionLayer(num_inputs=hidden_dim*2)
        self.final_linear=nn.Linear(hidden_dim*2,image_size*image_size*image_channels)
    def forward(self, x): 
        output_decoder=[]
        x=self.linear1(x)
        refined_memory = self.memorylayer1(x)
        x=self.ladder1(refined_memory,x)
        output_decoder.append(x)
        x=self.linear2(x)
        refined_memory2= self.memorylayer2(x)
        x=self.ladder2(refined_memory2,x)
        output_decoder.append(x)
        x=self.final_linear(x)                              
        return x,output_decoder
    
class MEMVAE(nn.Module):
    def __init__(self, n_levels, num_slots_list, slot_dim_list,latent_dim,hidden_dim,image_channels,image_size):
        super(MEMVAE,self).__init__()
        self.encoder=Encoder(num_features=image_size*image_size*image_channels, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder=Decoder(n_levels, num_slots_list, slot_dim_list,latent_dim,hidden_dim,image_channels,image_size)
        
    def forward_loss(self,image):
        image = image.view(image.size(0), -1) 
        z,mu, logvar,output_encoder=self.encoder(image)
        x_reconstruct,output_decoder=self.decoder(z)
        loss=0
        output1=output_encoder[0]
        output2=output_encoder[1]
        output3=output_decoder[0]
        output4=output_decoder[1]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        mse_loss = 0
        mse_loss = nn.functional.mse_loss(output3, output2, reduction='sum')+nn.functional.mse_loss(output4, output1, reduction='sum')+nn.functional.mse_loss(x_reconstruct,image,reduction='sum')
        loss=kl_loss+mse_loss
        return loss
    def forward(self, image):
        image = image.view(image.size(0), -1) 
        z,mu, logvar,output_encoder=self.encoder(image)
        x_reconstruct,output_encoder=self.decoder(z)
        loss=self.forward_loss(image)
        return x_reconstruct,loss