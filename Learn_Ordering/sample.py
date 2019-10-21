

from plotting_utils import * 




def make_sample_plot(model):

    samp_images = []

    # SAMPLE MODEL
    input_ = torch.zeros_like(img).cuda()
    allseenpixels_mask = torch.zeros(B,1,112*112).cuda()

    k_sample = 1
    n_steps_sample = 5

    for i_order in range(n_steps_sample):

        # mean, logsd, NLL_pred = model(input_)
        means, logsds, mixture_weight = model(input_)
        means = torch.clamp(means, min=0., max=1.)
        logsds = torch.clamp(logsds, min=-4.5, max=1.)

        dist = Mixture_Dist(k=4, means=means, logsds=logsds, mixture_weights=mixture_weight)


        mean = dist.get_mean()
        var = dist.get_var()
        sd =  var**(.5) #torch.sqrt(var)
        logsd = log_clamp( sd )

        logsd_pixelavg = torch.mean(logsd, dim=1)


        if i_order != n_steps_sample-1:

            # Get top error pixels - ie defining sampling order
            NLL_pred_scaled = logsd_pixelavg - torch.min(logsd_pixelavg) +.001  #shift so 0 is lowest
            NLL_pred_scaled = NLL_pred_scaled * (1.-allseenpixels_mask.view(B,112,112))  #remove the pixels that are revealed
            NLL_pred_scaled = NLL_pred_scaled.view(B,-1)

            values, indices = torch.topk(NLL_pred_scaled, k=k_sample*5, dim=1, largest=True, sorted=True) 

            # Take random set of k
            idx = torch.randperm(k_sample*5) 
            indices = indices[:,idx]
            indices = indices[:,:k_sample]

            # Make a mask out of it 
            sampled_pixels_mask = torch.zeros_like(NLL_pred_scaled).cuda()
            sampled_pixels_mask.scatter_(1,indices,1)



            samp = dist.sample()


            new_pixels = samp * sampled_pixels_mask.view(B,1,112,112)

            # Accumulate mask and mask the image
            allseenpixels_mask = allseenpixels_mask.view(B,112*112) + sampled_pixels_mask
            # input_ = (img * allseenpixels_mask.view(B,1,112,112)).detach()
            input_ = input_ + new_pixels


            samp_images.append(torch.clamp(input_, min=1e-5, max=1-1e-5))



        else:

            samp = dist.sample()

            new_pixels = samp * (1.-allseenpixels_mask.view(B,1,112,112))

            sampled_image = input_ + new_pixels

            samp_images.append(torch.clamp(sampled_image, min=1e-5, max=1-1e-5))


    



