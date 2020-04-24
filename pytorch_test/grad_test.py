

import torch



# # x = torch.rand(5, requires_grad=True)
# x = torch.ones(5, requires_grad=True)
# y = (x**2).sum()
# g1 = torch.autograd.grad(y,x,torch.ones(5), retain_graph=True)
# # g2 = torch.autograd.grad(y,x,None, retain_graph=True)
# # g1[0]==g2[0]
# # tensor([ 1,  1,  1,  1,  1], dtype=torch.uint8)
# print (g1)

# # so grad_outputs which is ones here NEEDS to be same size as y, or else it makes no sense


# x = torch.ones(5, requires_grad=True)
# # x = torch.rand(5, requires_grad=True)
# y = (x**2).sum()
# # g1 = torch.autograd.grad(y,x,torch.ones(5), retain_graph=True)
# g2 = torch.autograd.grad(y,x,None, retain_graph=True)
# print (g2)


# # x = torch.rand(5, requires_grad=True)
# x = torch.ones(5, requires_grad=True)
# y = (x**2).sum()
# g1 = torch.autograd.grad(torch.sum(y* torch.ones(5)), x,retain_graph=True)
# # g2 = torch.autograd.grad(y,x,None, retain_graph=True)
# # g1[0]==g2[0]
# # tensor([ 1,  1,  1,  1,  1], dtype=torch.uint8)
# print (g1)






# THIS ARE THE THREE EXAMPLES NEEDED TO UNERSTAND grad_outputs

x = torch.ones(5, requires_grad=True)
y = (x**2).sum()
g1 = torch.autograd.grad(y, x)
print (g1)

x = torch.ones(5, requires_grad=True)
y = (x**2)
g1 = torch.autograd.grad(y, x, torch.ones(5))
print (g1)

x = torch.ones(5, requires_grad=True)
y = (x**2)
g1 = torch.autograd.grad((y*torch.ones(5)).sum(), x)
print (g1)














