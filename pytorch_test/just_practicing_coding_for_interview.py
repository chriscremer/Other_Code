

# Logistic regression 

X = [N,D]
y = [N,1]


class Model():

    def __init__(self):

    	self.W = param([D,1])
    	self.b = param([D,1])
        

    def predict(x):

    	return sigmoid(x*w + b)


m = Model()

for step in range(max_steps):

	x,y = get_batch()

	pred = m.predict(x)

	loss = loss_func(pred, y)

	optimizer.zero_grad()
	loss.backwards()
	optimizer.step()


