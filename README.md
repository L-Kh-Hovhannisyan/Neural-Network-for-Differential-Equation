## Ordinary differential equation

We will start with simple ordinary differential equation (ODE) in the form of <br>
      <img 
      src="https://miro.medium.com/max/315/1*pvjvF0Q7YZa58BwFwCf77w.png"
      alt="html5" width="150" height="50" /> 

We are interested in finding a numerical solution on a grid, approximating it with some neural network architecture. In this article we will use very simple neural architecture that consists of a single input neuron (or two for 2D problems), one hidden layer and one output neuron to predict value of a solution in exact point on a grid.
The main question is how to transform equation integration problem in optimization one, e.g. minimizing the error between analytical (if it exists) and numerical solution, taking into account initial (IC) and boundary (BC) conditions. In paper (1) we can see that problem is transformed into the following system of equations:
<br>
<img 
      src="https://miro.medium.com/max/700/1*1oHXOKs3nGmq1mL6HcFlOg.png"
      alt="html5" width="355" height="60" /> 

In the proposed approach the trial solution Ψt employs a feedforward neural network and the parameters p correspond to the weights and biases of the neural architecture. In this work we omit biases for simplicity. We choose a form for the trial function Ψt(x) such that by construction satisfies the BCs. This is achieved by writing it as a sum of two terms:
<br>
<img 
      src="https://miro.medium.com/max/572/1*BVdGC_YhEIBrbJeG5NOIsA.png"
      alt="html5" width="250" height="40" /> 
<br>
where N(x, p) is a neural network of arbitrary architecture, weights of wich should be learnt to approximate the solution. For example in case of ODE, the trial solution will look like:
<br>
<img 
      src="https://miro.medium.com/max/427/1*yEQdMhnQ8idkk6XgYKVjsQ.png"
      alt="html5" width="250" height="50" /> 

And particular minimization problem to be solved is:

<img 
      src="https://miro.medium.com/max/672/1*ASrLCfwy6oZ9Y57xEmzhEQ.png"
      alt="html5" width="355" height="60" /> 

As we see, to minimize the error we need to calculate derivative of Ψt(x), our trial solution which contains neural network and terms that contain boundary conditions. In the paper (1) there is exact formula for NN derivatives, but whole trial solution can be too big to take derivatives by hand and hard-code them. We will use more elegant solution later, but for the first time we can code it:

```
def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])
def d_neural_network_dx(W, x, k=1):
    return np.dot(np.dot(W[1].T, W[0].T**k), sigmoid_grad(x))
def loss_function(W, x):
    loss_sum = 0.
    for xi in x:
        net_out = neural_network(W, xi)[0][0]
        psy_t = 1. + xi * net_out
        d_net_out = d_neural_network_dx(W, xi)[0][0]
        d_psy_t = net_out + xi * d_net_out
        func = f(xi, psy_t)       
        err_sqr = (d_psy_t - func)**2
        loss_sum += err_sqr
    return loss_sum
```

And optimization process, that basically is simple gradient descent… But wait, for gradient descent we need a derivative of solutions with respect to the weights, and we didn’t code it. Exactly. For this we will use modern tool for taking derivatives in so called “automatic differentiation” way — Autograd. It allows to take derivatives of any order of particular functions very easily and doesn’t require to mess with epsilon in finite difference approach or to type large formulas for symbolic differentiation software (MathCad, Mathematica, SymPy):
    
```
    W = [npr.randn(1, 10), npr.randn(10, 1)]
lmb = 0.001
for i in range(1000):
    loss_grad =  grad(loss_function)(W, x_space)

    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]
```
    
 Let’s try this on the following problem:

<img 
      src="https://miro.medium.com/max/655/1*OWbwgYIEU0dhVWmpug9QNw.png"
      alt="html5" width="355" height="60" /> 
    
We set up a grid [0, 1] with 10 points on it, BC is Ψ(0) = 1.
Result of training neural network for 1000 iterations with final mean squared error (MSE) of 0.0962 you can see on the image:

![equation](https://miro.medium.com/max/523/1*bYSwVxHdsrbSyFfYjwIcqg.png)

Just for fun I compared NN solution with finite differences one and we can see, that simple neural network without any parameters optimization works already better. Full code you can find here.

### Second order differential equation
Now we can go further and extend our solution to second-order equations:

<img 
      src="https://miro.medium.com/max/421/1*Ns0Cn2_BQee_m1pAJSZM2A.png"
      alt="html5" width="180" height="43" /> 
      <br>
      
that can have following trial solution (in case of two-point Dirichlet conditions)
<br>

<img 
      src="https://miro.medium.com/max/700/1*StrdqlwYgvY3iIQaAVVeFA.png"
      alt="html5" width="255" height="50" /> 

Taking derivatives of Ψt is getting harder and harder, so we will use Autograd more often:

```
def psy_trial(xi, net_out):
    return xi + xi**2 * net_out
psy_grad = grad(psy_trial)
psy_grad2 = grad(psy_grad)
def loss_function(W, x):
    loss_sum = 0.
    
    for xi in x:
        net_out = neural_network(W, xi)[0][0]
        net_out_d = grad(neural_network_x)(xi)
   
        psy_t = psy_trial(xi, net_out)
        
        gradient_of_trial = psy_grad(xi, net_out)
        second_gradient_of_trial = psy_grad2(xi, net_out)
        
        func = f(xi, psy_t, gradient_of_trial)
        
        err_sqr = (second_gradient_of_trial - func)**2
        loss_sum += err_sqr
        
    return loss_sum
 ```
 
 After 100 iterations and with MSE = 1.04 we can obtain following result of next equation:
 
 <img 
      src="https://miro.medium.com/max/349/1*rQbWISu5YO1TK6ypqEr8Tw.png"
      alt="html5" width="205" height="55" /> 
 
 ![equation](https://miro.medium.com/max/523/1*-wt5d2CN5xjBQwd_V-8W0w.png)
 
 You can get full code of this example from here.
### Partial differential equation
The most interesting processes are described with partial differential equations (PDEs), that can have the following form:
