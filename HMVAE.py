import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BrainInspiredMemoryLayer(nn.Module):
    """
    BrainInspiredMemoryLayer implements a memory module inspired by brain regions.
    
    This module simulates memory processing across multiple brain regions and levels.
    Each region has its own memory slots and controller, and a cross-region attention
    mechanism is applied to fuse information from all regions.
    
    Args:
        input_dim (int): Dimension of the input.
        n_levels (int): Number of hierarchical levels.
        n_regions (int): Number of brain regions.
        region_slot_config (list): List containing tuples of (num_slots_list, slot_dim_list) 
                                   for each brain region.
        act (nn.Module, optional): Activation function (default: nn.Sigmoid()).
        softmax_temperature (float, optional): Temperature for softmax scaling (default: 1.0).
        unified_dim (int, optional): Target unified dimension for projection (default: 64).
    """
    def __init__(self, input_dim, n_levels, n_regions, 
                 region_slot_config, act=nn.Sigmoid(), 
                 softmax_temperature=1.0, unified_dim=64):
        super(BrainInspiredMemoryLayer, self).__init__()
        self.n_regions = n_regions
        self.n_levels = n_levels
        self.unified_dim = unified_dim  # Target unified dimension for projection
        self.region_slot_config = region_slot_config

        # Define a nested module for the memory level of each region
        class RegionMemoryLevel(nn.Module):
            """
            A memory module for a single region at a specific hierarchical level.
            
            Args:
                num_slots (int): Number of memory slots in this region.
                slot_dim (int): Dimensionality of each memory slot.
                act (nn.Module): Activation function.
            """
            def __init__(self, num_slots, slot_dim, act):
                super().__init__()
                # Register memory and controller parameters
                self.memory = nn.Parameter(torch.Tensor(num_slots, slot_dim))
                self.controller = nn.Parameter(torch.Tensor(slot_dim, num_slots))
                self.bias = nn.Parameter(torch.zeros(num_slots))
                # Residual MLP for processing the aggregated memory
                self.res_mlp = nn.Sequential(
                    nn.Linear(slot_dim, slot_dim),
                    act,
                    nn.Linear(slot_dim, slot_dim)
                )
                # Initialize parameters using Xavier initialization
                nn.init.xavier_normal_(self.memory)
                nn.init.xavier_normal_(self.controller)
                
            def forward(self, x):
                # Forward pass logic will be handled in the outer layer.
                return x  # Placeholder

        # Create projectors for converting each region's output to a unified dimension
        self.region_projectors = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(region_slot_config[r][1][level], unified_dim)
                for level in range(n_levels)
            ]) for r in range(n_regions)
        ])
        # Create projections for each region at each level (processing features from the previous level)
        self.region_projections = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(
                    input_dim if level == 0 else region_slot_config[r][1][level-1],  # input from previous level
                    region_slot_config[r][1][level]  # current level slot dimension
                )
                for level in range(n_levels)
            ]) for r in range(n_regions)
        ])
        # Reverse projectors to map unified dimension back to original slot dimensions
        self.reverse_projectors = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(unified_dim, region_slot_config[r][1][level])
                for level in range(n_levels)
            ]) for r in range(n_regions)
        ])
        # Build memory structures for each brain region
        self.region_memories = nn.ModuleList()
        for region_idx in range(n_regions):
            num_slots_list, slot_dim_list = region_slot_config[region_idx]
            region_levels = nn.ModuleList()
            # Build a hierarchical memory structure for each region
            for level in range(n_levels):
                region_levels.append(
                    RegionMemoryLevel(
                        num_slots=num_slots_list[level],
                        slot_dim=slot_dim_list[level],
                        act=act
                    )
                )
            self.region_memories.append(region_levels)
        
        # Cross-region attention layer for fusing information from different brain regions
        self.cross_region_attn = nn.ModuleList([
            nn.Linear(unified_dim, n_regions)
            for _ in range(n_levels)
        ])
        
        # Final fusion layer to merge outputs from all regions and project back to original input dimension
        self.output_fusion = nn.Linear(
            n_regions * unified_dim,  # Concatenated unified features from all regions
            input_dim                 # Final output dimension matches the original input
        )
        
        self.act = act
        self.temp = softmax_temperature
        
        # Final projection for each region based on the last level's slot dimension
        self.final_proj = nn.ModuleList([
            nn.Linear(region_slot_config[r][1][-1], self.unified_dim)
            for r in range(self.n_regions)
        ])

    def forward(self, x):
        """
        Forward pass for the BrainInspiredMemoryLayer.
        
        The process includes three main stages:
        1. Independent processing per brain region (using memory attention and residual MLP).
        2. Dynamic projection to a unified dimension and cross-region attention fusion.
        3. Reverse projection back to the original region-specific dimensions.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].
        
        Returns:
            Tensor: Output tensor after memory and fusion operations.
        """
        batch_size = x.size(0)
        
        # Initialize each region's features by projecting the input into region-specific dimensions
        region_features = [
            self.region_projections[r][0](x) 
            for r in range(self.n_regions)
        ]
        
        # Process through each hierarchical level
        for level in range(self.n_levels):
            # Stage 1: Process features within each region independently.
            if level > 0:
                region_features = [
                    self.region_projections[r][level](region_features[r])
                    for r in range(self.n_regions)
                ]
            processed_regions = []
            for r in range(self.n_regions):
                layer = self.region_memories[r][level]
                current_slot_dim = self.region_slot_config[r][1][level]
                
                # Check that the input dimension matches the expected slot dimension
                assert region_features[r].shape[-1] == current_slot_dim, \
                    f"Level {level} region {r} input shape {region_features[r].shape} does not match slot_dim {current_slot_dim}"
                
                # Compute attention over memory slots
                attn = F.softmax(
                    (region_features[r] @ layer.controller + layer.bias) / self.temp,
                    dim=1
                )  # Shape: [B, num_slots]
                
                # Aggregate memory based on attention weights
                aggregated = attn @ layer.memory  # Shape: [B, slot_dim]
                residual = layer.res_mlp(aggregated)
                processed = region_features[r] + residual
                processed_regions.append(self.act(processed))
            
            # Stage 2: Project region outputs to unified dimension and fuse via cross-region attention.
            projected_regions = [
                self.region_projectors[r][level](processed_regions[r]) 
                for r in range(self.n_regions)
            ]
            
            # Stack projected features and compute cross-region attention
            all_regions = torch.stack(projected_regions, dim=1)  # Shape: [B, R, unified_dim]
            cross_attn_logits = self.cross_region_attn[level](all_regions.mean(dim=1))
            cross_attn = F.softmax(cross_attn_logits / self.temp, dim=1) 
            cross_aggregated = torch.einsum('br,brd->bd', cross_attn, all_regions)
            
            # Stage 3: Reverse projection back to region-specific dimensions with skip connections.
            for r in range(self.n_regions):
                reverse_proj = self.reverse_projectors[r][level](cross_aggregated)
                region_features[r] = processed_regions[r] + reverse_proj
        
        # Concatenate the final projected outputs from all regions and fuse them to the original input dimension.
        concatenated = torch.cat([
            self.final_proj[r](region_features[r]) 
            for r in range(self.n_regions)
        ], dim=1)
        return self.output_fusion(concatenated)


class LadderCompositionLayer(nn.Module):
    """
    LadderCompositionLayer refines activations by composing input features and latent variables.
    
    This module uses learnable parameters (a1, a2, a3, a4, c1, c2, c3, c4) to compute a refined 
    estimate of the latent variable via a non-linear transformation.
    
    Args:
        num_inputs (int): Number of input features.
        nonlinearity (nn.Module, optional): Intermediate activation function (default: nn.Sigmoid()).
        nonlinearity_final (nn.Module, optional): Final activation function (default: nn.ReLU()).
    """
    def __init__(self, num_inputs, nonlinearity=nn.Sigmoid(), nonlinearity_final=nn.ReLU()):
        super(LadderCompositionLayer, self).__init__()
        self.num_inputs = num_inputs
        self.nonlinearity = nonlinearity
        self.nonlinearity_final = nonlinearity_final

        # Learnable parameters for combining latent and upward activations
        self.a1 = nn.Parameter(torch.zeros(num_inputs))
        self.a2 = nn.Parameter(torch.ones(num_inputs))
        self.a3 = nn.Parameter(torch.zeros(num_inputs))
        self.a4 = nn.Parameter(torch.zeros(num_inputs))

        self.c1 = nn.Parameter(torch.zeros(num_inputs))
        self.c2 = nn.Parameter(torch.ones(num_inputs))
        self.c3 = nn.Parameter(torch.zeros(num_inputs))
        self.c4 = nn.Parameter(torch.zeros(num_inputs))

        # Fixed parameter (not learned) for bias
        self.b1 = nn.Parameter(torch.zeros(num_inputs), requires_grad=False)

    def forward(self, u, z_lat):
        """
        Forward pass to compute the refined latent variable.
        
        Combines the upward activation (u) and the latent variable (z_lat) with learned scaling and bias.
        
        Args:
            u (Tensor): Upward activation input.
            z_lat (Tensor): Latent variable input.
        
        Returns:
            Tensor: Refined latent representation after non-linear transformation.
        """
        sigval = self.c1 + self.c2 * z_lat
        sigval += self.c3 * u + self.c4 * z_lat * u
        sigval = self.nonlinearity(sigval)
        z_est = self.a1 + self.a2 * z_lat + self.b1 * sigval
        z_est += self.a3 * u + self.a4 * z_lat * u
        return self.nonlinearity_final(z_est)   


class Encoder(nn.Module):
    """
    Encoder module that maps input images to a latent space.
    
    The encoder uses a series of linear layers with batch normalization and ReLU activations.
    It outputs both the latent representation and intermediate activations used for ladder composition.
    
    Args:
        num_features (int): Flattened image feature dimension.
        hidden_dim (int): Dimension of hidden layers.
        latent_dim (int): Dimension of the latent space.
    """
    def __init__(self, num_features, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(num_features, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.latent = nn.Linear(hidden_dim, latent_dim)
        # Additional layers to generate mean and log variance for reparameterization
        self.get_mu = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.get_logvar = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def _reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from a Gaussian distribution.
        
        Args:
            mu (Tensor): Mean of the latent Gaussian.
            logvar (Tensor): Log variance of the latent Gaussian.
            
        Returns:
            Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, image):
        """
        Forward pass of the encoder.
        
        Args:
            image (Tensor): Flattened image tensor.
            
        Returns:
            tuple: (latent vector, mean, log variance, list of intermediate outputs)
        """
        output = []
        x = self.linears[0](image)
        x = self.linears[1](x)
        x = self.linears[2](x)
        output.append(x)
        temp = x
        x = self.linears[3](x)
        x = self.linears[4](x)
        x = self.linears[5](x)
        output.append(x)
        x = self.latent(x)
        mu = self.get_mu(temp)
        logvar = self.get_logvar(temp)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar, output
 

class Decoder(nn.Module):
    """
    Decoder module that reconstructs the image from the latent representation.
    
    The decoder uses two memory layers followed by ladder composition layers and linear layers to reconstruct the image.
    
    Args:
        n_levels (int): Number of hierarchical levels.
        n_regions (int): Number of brain regions.
        region_slot_config (list): Memory configuration for each brain region.
        latent_dim (int): Dimension of the latent space.
        hidden_dim (int): Dimension of hidden layers.
        image_channels (int): Number of channels in the output image.
        image_size (int): Height/width of the output image.
    """
    def __init__(self, n_levels, n_regions, region_slot_config, latent_dim, hidden_dim, image_channels, image_size):
        super(Decoder, self).__init__()
        self.memorylayer1 = BrainInspiredMemoryLayer(hidden_dim, n_levels, n_regions, region_slot_config, act=nn.Sigmoid(), softmax_temperature=1.0)
        self.memorylayer2 = BrainInspiredMemoryLayer(hidden_dim * 2, n_levels, n_regions, region_slot_config, act=nn.Sigmoid(), softmax_temperature=1.0)
        self.linear1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
        )
        self.ladder1 = LadderCompositionLayer(num_inputs=hidden_dim)
        self.ladder2 = LadderCompositionLayer(num_inputs=hidden_dim * 2)
        self.final_linear = nn.Linear(hidden_dim * 2, image_size * image_size * image_channels)
        
    def forward(self, x):
        """
        Forward pass of the decoder.
        
        Args:
            x (Tensor): Latent vector.
            
        Returns:
            tuple: (reconstructed image, list of intermediate outputs)
        """
        output_decoder = []
        x = self.linear1(x)
        refined_memory = self.memorylayer1(x)
        x = self.ladder1(refined_memory, x)
        output_decoder.append(x)
        x = self.linear2(x)
        refined_memory2 = self.memorylayer2(x)
        x = self.ladder2(refined_memory2, x)
        output_decoder.append(x)
        x = self.final_linear(x)
        return x, output_decoder
    

class HMVAE(nn.Module):
    """
    HMVAE is a Hierarchical Memory Variational Autoencoder that combines the encoder and decoder.
    
    The model uses the encoder to compress the input image into a latent representation, and the decoder 
    reconstructs the image using memory-inspired layers and ladder composition for refinement.
    
    Args:
        n_levels (int): Number of hierarchical levels.
        n_regions (int): Number of brain regions.
        region_slot_config (list): Memory slot configuration per brain region.
        latent_dim (int): Dimension of the latent space.
        hidden_dim (int): Dimension of hidden layers.
        image_channels (int): Number of channels in the input/output images.
        image_size (int): Height/width of the input/output images.
    """
    def __init__(self, n_levels, n_regions, region_slot_config, latent_dim, hidden_dim, image_channels, image_size):
        super(HMVAE, self).__init__()
        self.encoder = Encoder(num_features=image_size * image_size * image_channels, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(n_levels, n_regions, region_slot_config, latent_dim, hidden_dim, image_channels, image_size)
        
    def forward_loss(self, image):
        """
        Computes the VAE loss including the reconstruction loss and KL divergence.
        
        Args:
            image (Tensor): Input image tensor (flattened).
            
        Returns:
            Tensor: The combined loss.
        """
        image = image.view(image.size(0), -1) 
        z, mu, logvar, output_encoder = self.encoder(image)
        x_reconstruct, output_decoder = self.decoder(z)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        mse_loss = (nn.functional.mse_loss(output_decoder[0], output_encoder[1], reduction='sum') +
                    nn.functional.mse_loss(output_decoder[1], output_encoder[0], reduction='sum') +
                    nn.functional.mse_loss(x_reconstruct, image, reduction='sum'))
        loss = kl_loss + mse_loss
        return loss

    def forward(self, image):
        """
        Forward pass of the HMVAE.
        
        Args:
            image (Tensor): Input image tensor.
            
        Returns:
            tuple: (reconstructed image, loss)
        """
        image = image.view(image.size(0), -1) 
        z, mu, logvar, output_encoder = self.encoder(image)
        x_reconstruct, _ = self.decoder(z)
        loss = self.forward_loss(image)
        return x_reconstruct, loss
