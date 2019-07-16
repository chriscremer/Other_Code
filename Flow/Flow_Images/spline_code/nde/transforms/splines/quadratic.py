import math

import torch
from torch.nn import functional as F

import utils2
from nde import transforms

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3

# DEFAULT_MIN_BIN_WIDTH = .1
# DEFAULT_MIN_BIN_HEIGHT = 1e-3


def unconstrained_quadratic_spline(inputs,
                                   unnormalized_widths,
                                   unnormalized_heights,
                                   inverse=False,
                                   tail_bound=1.,
                                   tails='linear',
                                   min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                   min_bin_height=DEFAULT_MIN_BIN_HEIGHT):

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    num_bins = unnormalized_widths.shape[-1]

    # print (inputs.shape)
    # print (unnormalized_widths.shape)
    # print (unnormalized_heights.shape)
    # print (num_bins)
    # print (inside_interval_mask.shape)
    # # fasdf


    if tails == 'linear':
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
        assert unnormalized_heights.shape[-1] == num_bins - 1
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height
    )

    return outputs, logabsdet

def quadratic_spline(inputs,
                     unnormalized_widths,
                     unnormalized_heights,
                     inverse=False,
                     left=0., right=1., bottom=0., top=1.,
                     min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                     min_bin_height=DEFAULT_MIN_BIN_HEIGHT):
    if not inverse and (torch.min(inputs) < left or torch.max(inputs) > right):
        raise transforms.InputOutsideDomain()
    elif inverse and (torch.min(inputs) < bottom or torch.max(inputs) > top):
        raise transforms.InputOutsideDomain()

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    unnorm_heights_exp = torch.exp(unnormalized_heights)

    if unnorm_heights_exp.shape[-1] == num_bins - 1:
        # Set boundary heights s.t. after normalization they are exactly 1.
        first_widths = 0.5 * widths[..., 0]
        last_widths = 0.5 * widths[..., -1]
        numerator = (0.5 * first_widths * unnorm_heights_exp[..., 0]
                     + 0.5 * last_widths * unnorm_heights_exp[..., -1]
                     + torch.sum(((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2)
                                 * widths[..., 1:-1], dim=-1))
        constant = numerator / (1 - 0.5 * first_widths - 0.5 * last_widths)
        constant = constant[..., None]
        unnorm_heights_exp = torch.cat([constant, unnorm_heights_exp, constant], dim=-1)

    unnormalized_area = torch.sum(((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2)
                                  * widths, dim=-1)[..., None]
    heights = unnorm_heights_exp / unnormalized_area
    heights = min_bin_height + (1 - min_bin_height) * heights

    bin_left_cdf = torch.cumsum(((heights[..., :-1] + heights[..., 1:]) / 2) * widths, dim=-1)
    bin_left_cdf[..., -1] = 1.
    bin_left_cdf = F.pad(bin_left_cdf, pad=(1, 0), mode='constant', value=0.0)

    bin_locations = torch.cumsum(widths, dim=-1)
    bin_locations[..., -1] = 1.
    bin_locations = F.pad(bin_locations, pad=(1, 0), mode='constant', value=0.0)

    if inverse:
        bin_idx = utils2.searchsorted(bin_left_cdf, inputs)[..., None]
    else:
        bin_idx = utils2.searchsorted(bin_locations, inputs)[..., None]

    input_bin_locations = bin_locations.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_left_cdf = bin_left_cdf.gather(-1, bin_idx)[..., 0]

    input_left_heights = heights.gather(-1, bin_idx)[..., 0]
    input_right_heights = heights.gather(-1, bin_idx+1)[..., 0]


    a = 0.5 * (input_right_heights - input_left_heights) * input_bin_widths
    b = input_left_heights * input_bin_widths
    c = input_left_cdf

    if inverse:
        c_ = c - inputs
        # if (c_!=c_).any():
        #     print ('nan c_')
        #     fadf 
        # if (a!=a).any():
        #     print ('nan a')
        #     fadf 
        # if (b!=b).any():
        #     print ('nan b')
        #     fadf 

        # print ('a', torch.min(a), torch.max(a))
        # print ('b', torch.min(b), torch.max(b))
        # print ('c_', torch.min(c_), torch.max(c_))

        # a is near 0, resulting in nan/inf
        # alpha = (-b + torch.sqrt(b.pow(2) - 4*a*c_)) / (2*a)



        bot = torch.clamp(torch.abs(2*a), min=1e-5)
        not_zero = 1.-(torch.abs(2*a) > 0.).float()
        # print ()
        # print ('bot', torch.min(bot), torch.max(bot))
        bot = (torch.sign(2*a)+not_zero) * bot
        # print ('bot', torch.min(bot), torch.max(bot))
        # print ('abs bot', torch.min(torch.abs(bot)), torch.max(torch.abs(bot)))
        # print ('2*a', torch.min(2*a), torch.max(2*a))

        alpha = (-b + torch.sqrt(b.pow(2) - 4*a*c_)) / bot

        # print ('(2*a)', torch.min((2*a)), torch.max((2*a)))


        # eeee = (-b + torch.sqrt(b.pow(2) - 4*a*c_)) # / (2*a)
        # print ('eeee', torch.min(eeee), torch.max(eeee))
        # if (eeee!=eeee).any():
        #     print ('nan eeee')
        #     fadf 

        # bbbb = 1. / (2.*a)
        # print ('bbbb', torch.min(bbbb), torch.max(bbbb))
        # if (bbbb!=bbbb).any():
        #     print ('nan bbbb')
        #     fadf 

        if (alpha!=alpha).any():
            print ('bot', torch.min(bot), torch.max(bot))
            print ('abs bot', torch.min(torch.abs(bot)), torch.max(torch.abs(bot)))
            # print (input_right_heights)
            # print (input_left_heights)
            # print (input_bin_widths)
            # dif = input_right_heights - input_left_heights
            # print ('dif', torch.min(dif), torch.max(dif))
            print ('nan alpha')
            fadf 


        outputs = alpha * input_bin_widths + input_bin_locations
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = -torch.log((alpha * (input_right_heights - input_left_heights)
                                + input_left_heights))
        if (outputs!=outputs).any():
            print ('nan here')
            fadf 
    else:
        alpha = (inputs - input_bin_locations) / input_bin_widths
        outputs = a * alpha.pow(2) + b * alpha + c
        outputs = torch.clamp(outputs, 0, 1)
        logabsdet = torch.log((alpha * (input_right_heights - input_left_heights)
                               + input_left_heights))



    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - math.log(top - bottom) + math.log(right - left)
        if (outputs!=outputs).any():
            print ('nan here122222')
            fadf 
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + math.log(top - bottom) - math.log(right - left)


    return outputs, logabsdet










